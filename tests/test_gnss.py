"""GNSS 三维基线联合平差：白化、Mahalanobis 粗差、协方差贡献、预演、版本。"""
import math

import numpy as np
import pytest

from app.adjustment import AdjustmentError, run_adjustment
from app.models import (
    AdjustmentRequest,
    Baseline,
    KnownPoint,
    Observation,
    PreviewRequest,
)
from app.preview import run_preview
from conftest import gnss_payload


def diag_cov(sxy=0.005, sh=0.010, corr=0.12):
    C = np.diag([sxy**2, sxy**2, sh**2]).astype(float)
    C[0, 1] = C[1, 0] = corr * sxy**2
    return C.tolist()


# ---------- 基本解算与白化正确性 ----------

def test_gnss_only_network_adjusts():
    req = AdjustmentRequest.model_validate(gnss_payload())
    res = run_adjustment(req)
    st = {s["name"]: s for s in res["stations"]}
    assert st["G02"]["x"] == pytest.approx(500.0, abs=1e-6)
    assert st["G02"]["y"] == pytest.approx(0.0, abs=1e-6)
    assert st["G02"]["h"] == pytest.approx(100.5, abs=1e-6)
    assert st["G03"]["x"] == pytest.approx(1000.0, abs=1e-6)
    assert st["G04"]["y"] == pytest.approx(-500.0, abs=1e-6)
    # 4 条基线 ×3 行 = 12，参数 3 站 ×3 = 9，dof = 3
    stats = res["statistics"]
    assert stats["observations_m"] == 12
    assert stats["baseline_rows"] == 12
    assert stats["scalar_rows"] == 0
    assert stats["parameters_n"] == 9
    assert stats["rank"] == 9
    assert stats["rank_deficiency"] == 0
    assert stats["degrees_of_freedom"] == 3
    assert stats["converged"] is True
    assert len(res["baselines"]) == 4


def test_whitening_matches_generalized_least_squares():
    """白化后的法方程必须与直接 GLS：N = Σ JᵀC⁻¹J 完全一致。"""
    req = AdjustmentRequest.model_validate(gnss_payload())
    res = run_adjustment(req)
    sol = res["_internal"]["sol"]

    N_direct = np.zeros_like(sol["N"])
    for b in res["_internal"]["bls"]:
        # 用末态坐标重建 J（只含端点差，线性模型，等于迭代内雅可比）
        J = np.zeros((3, sol["n"]))
        pidx = sol["pidx"]
        for end, sgn in ((b["to"], 1.0), (b["frm"], -1.0)):
            if end in pidx:
                J[0, pidx[end]["x"]] = sgn
                J[1, pidx[end]["y"]] = sgn
                J[2, pidx[end]["h"]] = sgn
        N_direct += b["weight"] * J.T @ np.linalg.solve(b["C"], J)
    assert np.allclose(sol["N"], N_direct, rtol=1e-10, atol=1e-12)


def test_correlation_changes_result_vs_independent_components():
    """把三分量当成互不相关（置非对角为 0）应得到不同的坐标与点位协方差。

    两条到 C 的基线，dx 与 dh 强相关；忽略相关性会错误地按独立分量折中，
    相关解则会把误差更多分配到与相关结构一致的方向。
    """
    def solve(rho):
        bls = []
        for bid, frm, xA, dhA in (
                ("b1", "A", 100.01, 0.02), ("b2", "B", -89.99, 0.00)):
            C = np.diag([0.005**2, 0.005**2, 0.01**2])
            # 两条基线的 dx-dh 相关性异号，法方程耦合不再抵消
            C[0, 2] = C[2, 0] = rho * 0.005 * 0.01 * (
                1.0 if bid == "b1" else -1.0)
            bls.append(Baseline(**{
                "id": bid, "from": frm, "to": "C",
                "dx": xA, "dy": 0.0, "dh": dhA,
                "covariance": C.tolist()}))
        req = AdjustmentRequest(
            known=[KnownPoint(name="A", x=0, y=0, h=100.0),
                   KnownPoint(name="B", x=190, y=0, h=100.0)],
            stations=["C"], baselines=bls, save=False)
        return run_adjustment(req)

    r_corr = solve(0.9)
    r_indep = solve(0.0)
    sc = r_corr["stations"][0]
    si = r_indep["stations"][0]
    assert abs(sc["x"] - si["x"]) > 1e-9 or abs(sc["h"] - si["h"]) > 1e-9
    assert abs(np.array(sc["covariance_xyz_m2"])[0, 2]) > 0
    assert abs(np.array(si["covariance_xyz_m2"])[0, 2]) < 1e-15


def test_gnss_point_has_3d_covariance_and_ellipsoid():
    from scipy import stats
    res = run_adjustment(AdjustmentRequest.model_validate(gnss_payload()))
    k2 = float(stats.chi2.ppf(0.95, 2))
    k3 = float(stats.chi2.ppf(0.95, 3))
    for s in res["stations"]:
        assert s["std_x_m"] > 0 and s["std_y_m"] > 0 and s["std_h_m"] > 0
        D3 = np.array(s["covariance_xyz_m2"])
        assert D3.shape == (3, 3)
        assert np.allclose(D3, D3.T)
        assert np.all(np.linalg.eigvalsh(D3) >= 0)
        ell = s["error_ellipsoid"]
        assert len(ell["semi_axes_m"]) == 3
        assert ell["semi_axes_m"][0] >= ell["semi_axes_m"][1] >= ell["semi_axes_m"][2]
        assert ell["chi2_scale_df3"] == pytest.approx(k3)
        assert ell["chi2_scale_df3"] > k2


# ---------- Mahalanobis 统计量与粗差 ----------

def _two_baselines_to_one_point(C1, C2, d1, d2, w1=1.0, w2=1.0):
    """A、B 已知相距 100 m，两条相关基线到同一待求点 C。"""
    bls = [
        Baseline(id="b1", **{"from": "A", "to": "C",
                             "dx": d1[0], "dy": d1[1], "dh": d1[2],
                             "covariance": np.asarray(C1).tolist(), "weight": w1}),
        Baseline(id="b2", **{"from": "B", "to": "C",
                             "dx": d2[0], "dy": d2[1], "dh": d2[2],
                             "covariance": np.asarray(C2).tolist(), "weight": w2}),
    ]
    req = AdjustmentRequest(
        known=[KnownPoint(name="A", x=0, y=0, h=100.0),
               KnownPoint(name="B", x=100, y=0, h=100.0)],
        stations=["C"], baselines=bls, save=False, outlier_threshold=3.0)
    res = run_adjustment(req)
    return {b["id"]: b for b in res["baselines"]}


def test_mahalanobis_uses_full_residual_covariance():
    """报告场景：dx 残差 3.5、残差方差 10.5 时 Mahalanobis 必须为 1.166667，

    而不是把排序后的特征值当 dx 方差得到的 3.5²/1 = 12.25（误报粗差）。
    """
    C = np.diag([21.0, 21.0, 21.0])
    # xhat = 100，v_b1 = -3.5；Cv = C/2 = 10.5·I
    b1 = _two_baselines_to_one_point(
        C, C, [103.5, 0.0, 0.0], [-3.5, 0.0, 0.0])["b1"]
    Cv = np.array(b1["residual_covariance_m2"])
    assert np.allclose(np.diag(Cv), 10.5)
    assert b1["mahalanobis_prior"] == pytest.approx(3.5**2 / 10.5, abs=1e-9)
    assert b1["mahalanobis_prior"] == pytest.approx(1.166667, abs=1e-6)
    assert b1["p_value"] == pytest.approx(0.761, abs=0.01)
    assert b1["p_value"] > 0.5
    assert b1["is_outlier"] is False
    # 分量检验必须按 dx/dy/dh 顺序取 Cv 对角元（sqrt(10.5)），而非特征值
    ctests = {c["component"]: c for c in b1["component_tests"]}
    assert ctests["dx"]["residual_standard_error_m"] == pytest.approx(
        math.sqrt(10.5), abs=1e-9)
    assert ctests["dx"]["w"] == pytest.approx(-3.5 / math.sqrt(10.5), abs=1e-9)
    assert ctests["dy"]["w"] == 0.0 and ctests["dh"]["w"] == 0.0


def test_non_default_weight_residual_covariance_positive():
    """weight=4 时 Cv = C₀/w − J·N⁻¹·Jᵀ 必须仍为正定的理论值，

    修复前多乘一次 w 会得到负方差（0.05 − 4×0.0471 < 0）。
    """
    # C1=0.2·I, w1=4 -> 有效 C1*=0.05·I；C2=0.8·I -> C2*=0.8·I
    # x 方向：Ninv_x = 1/(4/0.2 + 1/0.8) = 1/21.25
    ninv_x = 1.0 / (4.0 / 0.2 + 1.0 / 0.8)
    expect_cv = 0.05 - ninv_x
    b1 = _two_baselines_to_one_point(
        np.diag([0.2, 0.2, 0.2]), np.diag([0.8, 0.8, 0.8]),
        [100.0, 0.0, 0.0], [0.0, 0.0, 0.0], w1=4.0)["b1"]
    Cv = np.array(b1["residual_covariance_m2"])
    assert expect_cv == pytest.approx(0.002941, abs=1e-6)
    assert np.allclose(np.diag(Cv), expect_cv, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(Cv) > 0)
    assert b1["is_outlier"] is False
    # 一致观测下残差为 0，Mahalanobis 为 0
    assert b1["mahalanobis_prior"] == pytest.approx(0.0, abs=1e-12)


def test_correlated_covariance_preserved_in_mahalanobis():
    """含非零相关项时：Cv 必须保留非对角元，Mahalanobis 用完整 3×3 逆，

    结果与把三分量当独立（只用对角元）不同。
    """
    C0 = np.array([[4.0, 2.0, 0.0],
                   [2.0, 4.0, 1.0],
                   [0.0, 1.0, 9.0]])
    # 等协方差两条基线 -> xhat=103.1, v_b1 = [-0.1, 0, 0]；Cv = C0/2
    b1 = _two_baselines_to_one_point(
        C0, C0, [103.0, 0.0, 0.0], [3.2, 0.0, 0.0])["b1"]
    v = np.array(b1["residual_vector_m"])
    Cv = np.array(b1["residual_covariance_m2"])
    assert np.allclose(v, [0.1, 0.0, 0.0], atol=1e-12)
    assert Cv[0, 1] == pytest.approx(1.0, abs=1e-12)
    assert Cv[1, 2] == pytest.approx(0.5, abs=1e-12)
    maha_full = float(v @ np.linalg.solve(Cv, v))
    maha_diag = float(np.sum(v**2 / np.diag(Cv)))
    assert maha_full != pytest.approx(maha_diag, rel=1e-6)
    # 解析值：0.02·35/104
    assert maha_full == pytest.approx(0.02 * 35.0 / 104.0, abs=1e-12)
    assert b1["mahalanobis_prior"] == pytest.approx(maha_full, abs=1e-12)
    # 分量标准误取物理顺序对角元 sqrt(2)
    ctests = {c["component"]: c for c in b1["component_tests"]}
    assert ctests["dx"]["residual_standard_error_m"] == pytest.approx(
        math.sqrt(2.0), abs=1e-12)


def test_baseline_residual_fields():
    res = run_adjustment(AdjustmentRequest.model_validate(gnss_payload()))
    b = res["baselines"][0]
    assert set(b["residual_vector"]) == {"dx", "dy", "dh"}
    assert len(b["residual_vector_m"]) == 3
    assert b["residual_norm_m"] >= 0
    Cv = np.array(b["residual_covariance_m2"])
    assert Cv.shape == (3, 3)
    assert np.allclose(Cv, Cv.T)
    assert b["mahalanobis_prior"] is not None and b["mahalanobis_prior"] >= 0
    assert 0.0 <= b["p_value"] <= 1.0
    assert 0.0 <= b["redundancy_number"] <= 1.0
    assert b["leverage"] >= 0.0
    assert len(b["component_tests"]) == 3
    assert np.array(b["precision_contribution_m2"]).shape == (3, 3)
    # 信息矩阵（协方差贡献）正定
    assert np.all(np.linalg.eigvalsh(np.array(b["precision_contribution_m2"])) > 0)
    # 端点协方差缩减：每个自由端点有 3×3 块且为半正定
    names = {e["name"] for e in b["endpoint_covariance_contribution"]}
    assert names == {"G02"}   # b1: G01 已知，G02 待求
    e = b["endpoint_covariance_contribution"][0]
    assert np.array(e["covariance_reduction_m2"]).shape == (3, 3)
    assert e["trace_reduction_m2"] >= -1e-12
    assert e["positive_semidefinite"] is True


def _redundant_baseline_network(gross_id=None, gross_component=2, gross_m=0.05):
    """高冗余不对称基线网（6 条、dof=9），粗差基线的 Mahalanobis 应显著突出。"""
    P = {"A": (0.0, 0.0, 100.0), "B": (100.0, 0.0, 100.0),
         "C": (100.0, 100.0, 101.0), "D": (0.0, 100.0, 99.0)}
    edges = [("g1", "A", "B"), ("g2", "B", "C"), ("g3", "C", "D"),
             ("g4", "D", "A"), ("g5", "A", "C"), ("g6", "B", "D")]
    bls = []
    for bid, f, t in edges:
        d = [P[t][0] - P[f][0], P[t][1] - P[f][1], P[t][2] - P[f][2]]
        if bid == gross_id:
            d[gross_component] += gross_m
        C = (np.diag([0.005**2, 0.005**2, 0.01**2])).tolist()
        bls.append(Baseline(**{
            "id": bid, "from": f, "to": t,
            "dx": d[0], "dy": d[1], "dh": d[2], "covariance": C}))
    return AdjustmentRequest(
        known=[KnownPoint(name="A", x=0, y=0, h=100.0)],
        stations=["B", "C", "D"], baselines=bls, save=False,
        outlier_threshold=3.0)


def test_mahalanobis_detects_gross_baseline():
    res = run_adjustment(_redundant_baseline_network(gross_id="g2"))
    by_id = {b["id"]: b for b in res["baselines"]}
    assert by_id["g2"]["is_outlier"] is True
    others = [by_id[b]["mahalanobis_prior"] for b in by_id if b != "g2"]
    assert by_id["g2"]["mahalanobis_prior"] > 2.0 * max(others)
    assert any(s["observation_id"] == "g2" and s["kind"] == "gnss_baseline"
               for s in res["suspects"])
    wdh = next(c for c in by_id["g2"]["component_tests"] if c["component"] == "dh")
    assert abs(wdh["w"]) >= 3.0


def test_clean_baselines_no_outliers():
    res = run_adjustment(AdjustmentRequest.model_validate(gnss_payload()))
    assert all(not b["is_outlier"] for b in res["baselines"])
    assert res["suspects"] == []


def test_known_to_known_baseline_checked():
    """两端都是已知点的基线不产生参数，但其差仍作为校核条件参与 χ²。"""
    req = AdjustmentRequest(
        known=[KnownPoint(name="A", x=0, y=0, h=100.0),
               KnownPoint(name="B", x=100, y=0, h=100.0)],
        stations=[],
        baselines=[Baseline(**{
            "from": "A", "to": "B", "dx": 100.03, "dy": 0.0, "dh": 0.0,
            "covariance": diag_cov(0.01, 0.01)})],
        save=False,
    )
    res = run_adjustment(req)
    b = res["baselines"][0]
    assert res["statistics"]["parameters_n"] == 0
    assert res["statistics"]["degrees_of_freedom"] == 3
    assert b["is_outlier"] is True
    assert b["endpoint_covariance_contribution"] == []


# ---------- 协方差预检 ----------

def test_asymmetric_covariance_rejected():
    payload = gnss_payload()
    C = np.array(payload["baselines"][0]["covariance"])
    C[0, 1] += 0.01
    payload["baselines"][0]["covariance"] = C.tolist()
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(AdjustmentRequest.model_validate(payload))
    assert e.value.code == "covariance_not_symmetric"
    assert e.value.details["id"] == "b1"


def test_indefinite_covariance_rejected():
    payload = gnss_payload()
    # 对角元素全为正，但相关系数 >1：矩阵不定（负特征值）
    payload["baselines"][0]["covariance"] = [
        [1.0, 1.5, 0.0], [1.5, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(AdjustmentRequest.model_validate(payload))
    assert e.value.code == "covariance_not_positive_definite"
    assert "min_eigenvalue_m2" in e.value.details
    assert e.value.details["min_eigenvalue_m2"] < 0


def test_negative_diagonal_covariance_rejected():
    payload = gnss_payload()
    payload["baselines"][0]["covariance"] = [
        [1.0, 0.0, 0.0], [0.0, -1e-6, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(AdjustmentRequest.model_validate(payload))
    assert e.value.code == "covariance_not_positive_definite"


def test_bad_covariance_shape_rejected():
    payload = gnss_payload()
    payload["baselines"][0]["covariance"] = [[1, 0, 0], [0, 1, 0]]
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(AdjustmentRequest.model_validate(payload))
    assert e.value.code == "bad_covariance_shape"
    assert e.value.details["shape"] == [2, 3]


def test_baseline_unknown_endpoint():
    payload = gnss_payload()
    payload["baselines"][0]["to"] = "GHOST"
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(AdjustmentRequest.model_validate(payload))
    assert e.value.code == "unknown_endpoint"


def test_baseline_known_endpoint_requires_height():
    req = AdjustmentRequest(
        known=[KnownPoint(name="A", x=0, y=0),
               KnownPoint(name="B", x=100, y=0, h=100.0)],
        stations=["C"],
        baselines=[
            Baseline(**{"from": "A", "to": "C", "dx": 10, "dy": 0, "dh": 1,
                        "covariance": diag_cov()}),
            Baseline(**{"from": "B", "to": "C", "dx": -90, "dy": 0, "dh": 1,
                        "covariance": diag_cov()}),
        ],
        save=False,
    )
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(req)
    assert e.value.code == "baseline_endpoint_without_height"


def test_empty_request_rejected():
    req = AdjustmentRequest(
        known=[KnownPoint(name="A", x=0, y=0, h=100.0)],
        stations=["B"], save=False)
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(req)
    assert e.value.code == "empty_observation"


def test_baseline_self_loop():
    req = AdjustmentRequest(
        known=[KnownPoint(name="A", x=0, y=0, h=100.0)],
        stations=["B"], save=False,
        baselines=[Baseline(**{
            "from": "A", "to": "A", "dx": 0, "dy": 0, "dh": 0,
            "covariance": diag_cov()})],
        observations=[Observation(**{
            "from": "A", "to": "B", "azimuth": 0.0, "distance": 100.0,
            "dh": 0.0})],
    )
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(req)
    assert e.value.code == "self_loop"


# ---------- 单位换算 ----------

def test_baseline_km_units():
    payload = gnss_payload()
    for b in payload["baselines"]:
        for k in ("dx", "dy", "dh"):
            b[k] = b[k] / 1000.0
        b["covariance"] = (np.array(b["covariance"]) / 1e6).tolist()
        b["unit"] = "km"
    res = run_adjustment(AdjustmentRequest.model_validate(payload))
    st = {s["name"]: s for s in res["stations"]}
    assert st["G03"]["x"] == pytest.approx(1000.0, abs=1e-6)
    # 输入单位残差按 km 给出，内部残差按 m
    rv_km = res["baselines"][0]["residual_vector"]
    rv_m = res["baselines"][0]["residual_vector_m"]
    assert rv_km["dx"] * 1000.0 == pytest.approx(rv_m[0], abs=1e-12)


# ---------- 混合网（全站仪 + GNSS） ----------

def mixed_payload(**ov):
    """A、B 已知，C、D 待求；4 条全站仪闭合边 + 1 条相关基线 A->C。"""
    P = {"A": (0.0, 0.0, 100.0), "B": (100.0, 0.0, 100.0),
         "C": (100.0, 100.0, 101.0), "D": (0.0, 100.0, 99.0)}
    obs = []
    for i, (f, t) in enumerate(
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")], start=1):
        dx, dy = P[t][0] - P[f][0], P[t][1] - P[f][1]
        obs.append({"id": f"e{i}", "from": f, "to": t,
                    "azimuth": round(math.degrees(math.atan2(dy, dx)), 6),
                    "distance": math.hypot(dx, dy),
                    "dh": P[t][2] - P[f][2]})
    C = np.diag([0.01**2, 0.01**2, 0.02**2])
    C[0, 1] = C[1, 0] = 0.5e-4
    bl = [{"id": "gAC", "from": "A", "to": "C", "dx": 100.0, "dy": 100.0,
           "dh": 1.0, "covariance": C.tolist()}]
    payload = {
        "known": [{"name": "A", "x": 0, "y": 0, "h": 100.0},
                  {"name": "B", "x": 100, "y": 0, "h": 100.0}],
        "stations": ["C", "D"],
        "observations": obs, "baselines": bl,
    }
    payload.update(ov)
    return payload


def test_mixed_network():
    res = run_adjustment(AdjustmentRequest.model_validate(mixed_payload()))
    stats = res["statistics"]
    # 标量边 4×3 = 12 行 + 基线 3 行 = 15；参数 2 站×3 = 6；dof 9
    assert stats["scalar_rows"] == 12
    assert stats["baseline_rows"] == 3
    assert stats["observations_m"] == 15
    assert stats["parameters_n"] == 6
    assert stats["degrees_of_freedom"] == 9
    st = {s["name"]: s for s in res["stations"]}
    assert st["C"]["x"] == pytest.approx(100.0, abs=1e-4)
    assert st["C"]["h"] == pytest.approx(101.0, abs=1e-3)
    assert len(res["observations"]) == 4 and len(res["baselines"]) == 1


def test_mixed_single_known_point_orientation_fixed_by_baseline():
    """只有 1 个已知点时，GNSS 基线提供方向，全站仪方位角可随之解算。"""
    payload = mixed_payload()
    payload["known"] = [payload["known"][0]]
    payload["stations"] = ["B", "C", "D"]
    res = run_adjustment(AdjustmentRequest.model_validate(payload))
    assert res["statistics"]["rank"] == 9
    st = {s["name"]: s for s in res["stations"]}
    assert st["B"]["x"] == pytest.approx(100.0, abs=1e-3)


# ---------- 预演 ----------

def test_preview_disable_baseline():
    req = AdjustmentRequest.model_validate(mixed_payload())
    base = run_adjustment(req)
    pv = run_preview(PreviewRequest(base=req, disable_baselines=["gAC"]))
    comp = pv["comparison"]
    assert comp["disabled_baselines"] == ["gAC"]
    assert comp["baseline_count"] == {"base": 1, "preview": 0}
    # 停用一条 3D 基线：少 3 个观测行
    assert comp["degrees_of_freedom"]["preview"] == \
        comp["degrees_of_freedom"]["base"] - 3
    assert comp["rank"] == {"base": 6, "preview": 6}
    # 坐标发生变化
    shifts = {s["name"]: s for s in comp["station_shifts"]}
    assert shifts["C"]["horizontal_shift_m"] >= 0.0


def test_preview_baseline_covariance_scale_moves_toward_ts_solution():
    req = AdjustmentRequest.model_validate(mixed_payload())
    # 让基线偏离全站仪网：dx 偏大 2 cm
    req.baselines[0].dx = 100.02
    base = run_adjustment(req.model_copy(deep=True))
    c_base = {s["name"]: s for s in base["stations"]}["C"]["x"]
    pv = run_preview(PreviewRequest(base=req, baseline_covariance_scale=10000.0))
    c_weak = {s["name"]: s for s in pv["comparison"]["station_shifts"]}["C"]["dx_m"]
    # 基线被大幅放宽后，C.x 应向纯全站仪解（恰为 100.0）回落
    ts_only = run_preview(PreviewRequest(
        base=req, disable_baselines=["gAC"]))
    c_ts = {s["name"]: s for s in ts_only["comparison"]["station_shifts"]}["C"]["dx_m"]
    assert abs(c_weak - c_ts) < abs(c_base - 100.0) + 1e-9
    assert pv["comparison"]["baseline_covariance_scale"] == 10000.0


def test_preview_per_baseline_covariance_scale():
    req = AdjustmentRequest.model_validate(mixed_payload())
    pv = run_preview(PreviewRequest(
        base=req, baseline_covariance_scales={"gAC": 9.0}))
    assert pv["comparison"]["baseline_covariance_scales"] == {"gAC": 9.0}
    # 自由度不变（只是调权）
    assert pv["comparison"]["degrees_of_freedom"]["preview"] == \
        pv["comparison"]["degrees_of_freedom"]["base"]


def test_preview_baseline_weight_zero_disables():
    req = AdjustmentRequest.model_validate(mixed_payload())
    pv = run_preview(PreviewRequest(
        base=req, baseline_weight_overrides={"gAC": 0.0}))
    assert pv["comparison"]["baseline_count"]["preview"] == 0


def test_preview_unknown_baseline_ids():
    req = AdjustmentRequest.model_validate(mixed_payload())
    pv = run_preview(PreviewRequest(
        base=req, disable_baselines=["nope"],
        baseline_covariance_scales={"zzz": 4.0}))
    assert set(pv["comparison"]["unknown_baseline_ids"]) == {"nope", "zzz"}
