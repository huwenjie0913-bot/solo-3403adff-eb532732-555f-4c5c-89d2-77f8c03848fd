"""API 端到端：GNSS 基线联合平差、预检错误码、版本复算、预演。"""
import numpy as np

from conftest import gnss_payload


def test_gnss_adjustment_and_version_roundtrip(client):
    payload = gnss_payload()
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["statistics"]["baselines"] == 4
    assert body["statistics"]["baseline_rows"] == 12
    assert body["statistics"]["degrees_of_freedom"] == 3
    assert body["statistics"]["rank"] == 9
    assert len(body["baselines"]) == 4
    b0 = body["baselines"][0]
    assert len(b0["residual_vector_m"]) == 3
    assert "mahalanobis_prior" in b0 and "p_value" in b0
    assert len(b0["precision_contribution_m2"]) == 3
    # 点位三维协方差
    st = {s["name"]: s for s in body["stations"]}
    assert len(st["G02"]["covariance_xyz_m2"]) == 3
    assert len(st["G02"]["error_ellipsoid"]["semi_axes_m"]) == 3
    # 含基线联合平差提示
    assert any("GNSS" in w and "Cholesky" in w for w in body["warnings"])

    sid = body["version"]["scheme_id"]
    vno = body["version"]["version_no"]
    # 版本中保存完整基线请求
    v = client.get(f"/api/v1/schemes/{sid}/versions/{vno}").json()
    assert len(v["request"]["baselines"]) == 4
    assert v["summary"]["baseline_count"] == 4
    # 按原参数复算
    rc = client.post(
        f"/api/v1/schemes/{sid}/versions/{vno}/recompute").json()
    assert len(rc["baselines"]) == 4
    assert rc["statistics"]["rank"] == 9
    assert rc["recomputed_from"]["request_parameters"]["baselines"][0]["covariance"]


def test_gnss_baseline_outlier_flagged(client):
    payload = gnss_payload(gross_baseline="b3", gross_component="dh", gross_m=0.30)
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    suspects = {s["observation_id"]: s for s in body["suspects"]}
    assert "b3" in suspects
    assert suspects["b3"]["kind"] == "gnss_baseline"
    b3 = next(b for b in body["baselines"] if b["id"] == "b3")
    assert b3["is_outlier"] is True
    assert b3["hint"]


def test_asymmetric_covariance_returns_400(client):
    payload = gnss_payload()
    payload["baselines"][1]["covariance"][0][2] += 0.01
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 400
    err = r.json()
    assert err["error"] == "covariance_not_symmetric"
    assert err["details"]["id"] == "b2"


def test_non_pd_covariance_returns_400(client):
    payload = gnss_payload()
    payload["baselines"][0]["covariance"] = [
        [1, 1.5, 0], [1.5, 1, 0], [0, 0, 1]]
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 400
    assert r.json()["error"] == "covariance_not_positive_definite"


def test_bad_covariance_shape_returns_400(client):
    payload = gnss_payload()
    payload["baselines"][0]["covariance"] = [[1, 0, 0]]
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 400
    assert r.json()["error"] == "bad_covariance_shape"


def _mixed_preview_payload():
    """2 已知点 + 全站仪闭合导线（A-B-C-D-A）+ 2 条 GNSS 基线，停用基线仍可解。"""
    import math
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
    C = np.diag([0.01**2, 0.01**2, 0.02**2]).tolist()
    baselines = [
        {"id": "g1", "from": "A", "to": "C", "dx": 100.0, "dy": 100.0,
         "dh": 1.0, "covariance": C},
        {"id": "g2", "from": "B", "to": "D", "dx": -100.0, "dy": 100.0,
         "dh": -1.0, "covariance": C},
    ]
    return {
        "name": "mixed-preview",
        "known": [{"name": "A", "x": 0, "y": 0, "h": 100.0},
                  {"name": "B", "x": 100, "y": 0, "h": 100.0}],
        "stations": ["C", "D"],
        "observations": obs, "baselines": baselines,
    }


def test_gnss_preview_disable_and_scale(client):
    payload = _mixed_preview_payload()
    r = client.post("/api/v1/preview", json={
        "base": payload, "disable_baselines": ["g1"],
    })
    assert r.status_code == 200, r.text
    comp = r.json()["comparison"]
    assert comp["baseline_count"] == {"base": 2, "preview": 1}
    assert comp["disabled_baselines"] == ["g1"]
    # 停用一条 3D 基线：少 3 个观测行
    assert comp["degrees_of_freedom"]["preview"] == \
        comp["degrees_of_freedom"]["base"] - 3
    assert comp["rank"] == {"base": 6, "preview": 6}
    # 点位位移有数值
    shifts = {s["name"]: s for s in comp["station_shifts"]}
    assert shifts["C"]["horizontal_shift_m"] is not None

    # 停用全部基线：全站仪闭合导线仍有 2 已知点，可解
    r_all = client.post("/api/v1/preview", json={
        "base": payload, "disable_baselines": ["g1", "g2"],
    })
    assert r_all.status_code == 200, r_all.text
    assert r_all.json()["comparison"]["baseline_count"] == {"base": 2, "preview": 0}

    # 整体缩放协方差：自由度/秩不变，σ0 改变
    r2 = client.post("/api/v1/preview", json={
        "base": payload, "baseline_covariance_scale": 100.0,
    })
    c2 = r2.json()["comparison"]
    assert c2["degrees_of_freedom"]["preview"] == c2["degrees_of_freedom"]["base"]
    assert c2["rank"] == {"base": 6, "preview": 6}
    assert c2["baseline_covariance_scale"] == 100.0

    # 未知基线 id 上报
    r3 = client.post("/api/v1/preview", json={
        "base": payload, "disable_baselines": ["ghost"]})
    assert "ghost" in r3.json()["comparison"]["unknown_baseline_ids"]


def test_gnss_km_units_endpoint(client):
    payload = gnss_payload()
    for b in payload["baselines"]:
        for k in ("dx", "dy", "dh"):
            b[k] = b[k] / 1000.0
        b["covariance"] = (np.array(b["covariance"]) / 1e6).tolist()
        b["unit"] = "km"
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 200, r.text
    st = {s["name"]: s for s in r.json()["stations"]}
    assert abs(st["G03"]["x"] - 1000.0) < 1e-6
