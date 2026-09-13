"""多期形变分析：位移恢复、显著性、阈值、两期/多期、基准比较、版本。"""
import math

import pytest

from app.adjustment import AdjustmentError
from app.deformation import run_deformation
from app.models import DeformationRequest
from conftest import deformation_payload


def _points_by_name(result):
    return {p["name"]: p for p in result["points"]}


# ---------- 引擎层 ----------

def test_multi_epoch_recovers_displacement():
    res = run_deformation(DeformationRequest.model_validate(deformation_payload()))
    assert res["status"] == "ok"
    assert res["reference_epoch"] == "E1"
    assert [e["epoch"] for e in res["epochs"]] == ["E1", "E2", "E3"]
    assert res["epochs"][0]["is_reference"] is True
    # 时间间隔：2026-01-01 -> 2026-05-01 = 120 天
    assert res["epochs"][2]["days_from_reference"] == pytest.approx(120.0)

    pts = _points_by_name(res)
    m1 = pts["M1"]
    # E3 期位移分量恢复真值 (+0.05, +0.02, +0.01)
    d_e3 = next(d for d in m1["displacements"] if d["epoch"] == "E3")
    assert d_e3["dx_m"] == pytest.approx(0.05, abs=5e-3)
    assert d_e3["dy_m"] == pytest.approx(0.02, abs=5e-3)
    assert d_e3["dh_m"] == pytest.approx(0.01, abs=5e-3)
    assert d_e3["d3d_m"] == pytest.approx(math.sqrt(0.003), abs=5e-3)
    # 速度：0.05 m / (120/365.25) 年
    years = 120.0 / 365.25
    assert d_e3["velocity"]["vx_m_per_year"] == pytest.approx(0.05 / years, rel=0.1)
    # 联合检验显著，p 值极小
    assert d_e3["test"]["df"] == 3
    assert d_e3["test"]["significant"] is True
    assert d_e3["test"]["p_value"] < 1e-3
    # 置信椭球三半轴递减且为正
    axes = d_e3["ellipsoid"]["semi_axes_m"]
    assert len(axes) == 3 and axes[0] >= axes[1] >= axes[2] > 0
    assert d_e3["ellipsoid"]["chi2_scale"] == pytest.approx(7.815, abs=1e-3)
    # 参与观测
    assert "E3-D1M1" in d_e3["observation_ids"]
    # 首个超阈值时刻
    assert m1["first_exceed_epoch"] == "E3"
    assert m1["first_exceed_time"].startswith("2026-05-01")
    assert m1["significant"] is True

    # M2 未位移：不显著、不超阈值
    m2 = pts["M2"]
    assert m2["significant"] is False
    assert m2["first_exceed_epoch"] is None
    assert m2["max_displacement_m"] < 0.02
    for d in m2["displacements"]:
        assert d["test"]["p_value"] > 0.01

    # 基准点：位移恒零、不做检验
    d1 = pts["D1"]
    assert d1["role"] == "datum"
    assert all(d["test"] is None for d in d1["displacements"])
    assert all(d["d3d_m"] == 0.0 for d in d1["displacements"])

    # 汇总
    assert res["summary"]["points_exceeding_threshold"] == ["M1"]
    assert res["summary"]["points_significant"] == ["M1"]


def test_two_epoch_selection():
    payload = deformation_payload()
    payload["epoch_selection"] = ["E3", "E1"]  # 乱序给出，应按时间排序
    res = run_deformation(DeformationRequest.model_validate(payload))
    assert [e["epoch"] for e in res["epochs"]] == ["E1", "E3"]
    m1 = _points_by_name(res)["M1"]
    assert len(m1["displacements"]) == 1
    d = m1["displacements"][0]
    assert d["epoch"] == "E3"
    assert d["dx_m"] == pytest.approx(0.05, abs=5e-3)


def test_two_epoch_selection_without_deformation():
    payload = deformation_payload()
    payload["epoch_selection"] = ["E1", "E2"]  # 两期均未位移
    res = run_deformation(DeformationRequest.model_validate(payload))
    m1 = _points_by_name(res)["M1"]
    assert m1["significant"] is False
    assert m1["first_exceed_epoch"] is None


def test_alternative_datum_comparison():
    payload = deformation_payload()
    payload["alternative_datum"] = ["D1", "M2"]
    res = run_deformation(DeformationRequest.model_validate(payload))
    comp = res["datum_comparison"]
    assert comp["primary_datum"] == ["D1", "D2"]
    assert comp["alternative_datum"] == ["D1", "M2"]
    # 对齐参数逐期给出
    assert len(comp["alignment_to_primary"]) == 3
    rows = {r["name"]: r for r in comp["points"]}
    # M1 在两种基准下结论一致：显著且超阈值
    assert rows["M1"]["primary"]["significant"] is True
    assert rows["M1"]["alternative"]["significant"] is True
    assert rows["M1"]["conclusion_changed"] is False
    # 替代基准下 M1 的位移仍应恢复到真值附近
    assert rows["M1"]["alternative"]["max_displacement_m"] == pytest.approx(
        math.sqrt(0.003), abs=5e-3)
    # 替代基准点 M2 在替代方案中是基准（位移为零、不显著）
    assert rows["M2"]["alternative"]["role"] == "datum"
    assert rows["M2"]["alternative"]["significant"] is False


def test_alternative_datum_station_only_at_from_end():
    """回归：原基准点 D2 转为待求点后只位于完整边的 from 端（网络中无
    任何 ->D2 的边），初值反向传播应使替代基准比较无需补测 D1->D2。"""
    payload = deformation_payload()
    # 确认回归场景：D2 在所有期中都不出现在任何边的 to 端
    for ep in payload["epochs"]:
        assert all(o["to"] != "D2" for o in ep["observations"])
    payload["alternative_datum"] = ["D1", "M2"]  # D2 变为待求点
    res = run_deformation(DeformationRequest.model_validate(payload))
    assert res["status"] == "ok"  # 修复前：400 no_initial_coordinates
    rows = {r["name"]: r for r in res["datum_comparison"]["points"]}
    assert rows["M1"]["alternative"]["significant"] is True
    # D2 在替代方案中是自由点，位移应与主基准结论一致（不显著）
    assert rows["D2"]["alternative"]["role"] == "monitor"
    assert rows["D2"]["alternative"]["significant"] is False


# ---------- 预检错误 ----------

def test_error_epoch_time_reversed():
    payload = deformation_payload()
    payload["epochs"][1]["time"] = "2026-07-01"  # 比 E3 还晚 -> 倒序
    with pytest.raises(AdjustmentError) as e:
        run_deformation(DeformationRequest.model_validate(payload))
    assert e.value.code == "epoch_time_order"
    assert e.value.details["previous"]["epoch"] == "E2"
    assert e.value.details["next"]["epoch"] == "E3"


def test_error_epoch_point_mismatch():
    payload = deformation_payload()
    # 删除 E2 中所有涉及 M2 的观测 -> E2 点集缺少 M2
    payload["epochs"][1]["observations"] = [
        o for o in payload["epochs"][1]["observations"]
        if "M2" not in (o["from"], o["to"])
    ]
    with pytest.raises(AdjustmentError) as e:
        run_deformation(DeformationRequest.model_validate(payload))
    assert e.value.code == "epoch_point_mismatch"
    assert e.value.details["missing_by_epoch"] == {"E2": ["M2"]}


def test_error_insufficient_datum_points():
    payload = deformation_payload()
    payload["datum_points"] = payload["datum_points"][:1]
    with pytest.raises(AdjustmentError) as e:
        run_deformation(DeformationRequest.model_validate(payload))
    assert e.value.code == "insufficient_datum_points"
    assert e.value.details["datum_count"] == 1


def test_error_datum_point_not_observed():
    payload = deformation_payload()
    payload["datum_points"].append({"name": "D9", "x": 0.0, "y": 0.0})
    with pytest.raises(AdjustmentError) as e:
        run_deformation(DeformationRequest.model_validate(payload))
    assert e.value.code == "datum_point_not_observed"
    assert e.value.details["points"] == ["D9"]


def test_error_bad_epoch_time():
    payload = deformation_payload()
    payload["epochs"][0]["time"] = "二零二六年一月"
    with pytest.raises(AdjustmentError) as e:
        run_deformation(DeformationRequest.model_validate(payload))
    assert e.value.code == "bad_epoch_time"


def test_error_epoch_selection_invalid():
    payload = deformation_payload()
    payload["epoch_selection"] = ["E1", "E2", "E3"]
    with pytest.raises(AdjustmentError) as e:
        run_deformation(DeformationRequest.model_validate(payload))
    assert e.value.code == "epoch_selection_invalid"

    payload["epoch_selection"] = ["E1", "E9"]
    with pytest.raises(AdjustmentError) as e:
        run_deformation(DeformationRequest.model_validate(payload))
    assert e.value.code == "epoch_selection_invalid"


def test_error_alternative_datum_invalid():
    payload = deformation_payload()
    payload["alternative_datum"] = ["D1", "GHOST"]
    with pytest.raises(AdjustmentError) as e:
        run_deformation(DeformationRequest.model_validate(payload))
    assert e.value.code == "alternative_datum_invalid"


# ---------- API 层 ----------

def test_api_deformation_saves_and_retrieves_version(client):
    r = client.post("/api/v1/deformations", json=deformation_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]["version_no"] == 1
    sid = body["version"]["scheme_id"]

    # 方案与版本列表
    schemes = client.get("/api/v1/deformation-schemes").json()["schemes"]
    assert any(s["name"] == "dam-monitor" for s in schemes)
    versions = client.get(f"/api/v1/deformation-schemes/{sid}/versions").json()
    assert len(versions["versions"]) == 1
    assert versions["versions"][0]["summary"]["reference_epoch"] == "E1"

    # 按编号复取：参数完整可追溯
    v = client.get(f"/api/v1/deformation-schemes/{sid}/versions/1").json()
    assert v["request"]["displacement_threshold"] == 0.02
    assert v["request"]["datum_points"][0]["name"] == "D1"
    assert len(v["request"]["epochs"]) == 3
    assert v["software"]["numpy"]
    assert v["summary"]["summary"]["points_significant"] == ["M1"]

    # 再次提交同名方案 -> version_no=2
    r2 = client.post("/api/v1/deformations", json=deformation_payload())
    assert r2.json()["version"]["version_no"] == 2


def test_api_deformation_recompute_two_epochs(client):
    r = client.post("/api/v1/deformations", json=deformation_payload())
    sid = r.json()["version"]["scheme_id"]

    # 覆盖为两期重算（不落库）
    rc = client.post(
        f"/api/v1/deformation-schemes/{sid}/versions/1/recompute",
        json={"epoch_selection": ["E1", "E3"]},
    )
    assert rc.status_code == 200, rc.text
    body = rc.json()
    assert body["recomputed_from"]["version_no"] == 1
    assert body["version"] is None
    assert [e["epoch"] for e in body["epochs"]] == ["E1", "E3"]
    m1 = _points_by_name(body)["M1"]
    assert len(m1["displacements"]) == 1
    assert m1["displacements"][0]["dx_m"] == pytest.approx(0.05, abs=5e-3)

    # 默认复算沿用原参数（连续多期），save=true 产生新版本
    rc2 = client.post(
        f"/api/v1/deformation-schemes/{sid}/versions/1/recompute?save=true")
    assert rc2.status_code == 200
    assert rc2.json()["version"]["version_no"] == 2
    assert len(rc2.json()["epochs"]) == 3

    # 不存在的版本 -> 404
    r404 = client.get(f"/api/v1/deformation-schemes/{sid}/versions/99")
    assert r404.status_code == 404


def test_api_deformation_validation_errors(client):
    payload = deformation_payload()
    payload["epochs"][2]["time"] = "2025-12-31"  # 时间倒序
    r = client.post("/api/v1/deformations", json=payload)
    assert r.status_code == 400
    assert r.json()["error"] == "epoch_time_order"

    payload = deformation_payload()
    payload["datum_points"] = payload["datum_points"][:1]
    r = client.post("/api/v1/deformations", json=payload)
    assert r.status_code == 400
    assert r.json()["error"] == "insufficient_datum_points"

    payload = deformation_payload()
    payload["epochs"][0]["observations"] = [
        o for o in payload["epochs"][0]["observations"]
        if "M2" not in (o["from"], o["to"])
    ]
    r = client.post("/api/v1/deformations", json=payload)
    assert r.status_code == 400
    assert r.json()["error"] == "epoch_point_mismatch"


def test_api_deformation_alternative_datum(client):
    payload = deformation_payload()
    payload["alternative_datum"] = ["D1", "M2"]
    r = client.post("/api/v1/deformations", json=payload)
    assert r.status_code == 200, r.text
    comp = r.json()["datum_comparison"]
    assert comp["conclusion_changed_points"] == []
    rows = {x["name"]: x for x in comp["points"]}
    assert rows["M1"]["primary"]["significant"] is True
    assert rows["M1"]["alternative"]["significant"] is True
