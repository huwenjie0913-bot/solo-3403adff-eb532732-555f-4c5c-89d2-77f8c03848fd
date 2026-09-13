"""引擎层：单位换算、平差、预检、粗差。"""
import math

import pytest

from app.adjustment import AdjustmentError, run_adjustment
from app.models import AdjustmentRequest, KnownPoint, Observation
from app.units import (
    angle_to_rad, distance_to_m, rad_to_deg, rad_to_dms, normalize_angle,
)
from conftest import square_traverse_payload, leveling_payload


# ---------- 单位 ----------

def test_dms_parsing():
    r = angle_to_rad("112-30-00")
    assert abs(rad_to_deg(r) - 112.5) < 1e-12
    r2 = angle_to_rad("112°30′15″")
    assert abs(rad_to_deg(r2) - (112 + 30 / 60 + 15 / 3600)) < 1e-12


def test_gon_conversion():
    assert abs(angle_to_rad(100.0, "gon") - math.pi / 2) < 1e-12


def test_distance_units():
    assert distance_to_m(1.0, "km") == pytest.approx(1000.0)
    assert distance_to_m(1.0, "ft") == pytest.approx(0.3048)


def test_dms_formatting_roundtrip():
    d = rad_to_dms(angle_to_rad("45-59-59.9995"))
    assert d["deg"] == 46 and d["minute"] == 0 and d["second"] == 0.0


# ---------- 基本平差 ----------

def test_square_traverse_adjusts_and_reduces_misclosure():
    payload = square_traverse_payload()
    req = AdjustmentRequest.model_validate(payload)
    res = run_adjustment(req)

    by_name = {s["name"]: s for s in res["stations"]}
    assert by_name["C"]["x"] == pytest.approx(1000.0, abs=2e-3)
    assert by_name["C"]["y"] == pytest.approx(1000.0, abs=2e-3)
    assert by_name["D"]["x"] == pytest.approx(0.0, abs=2e-3)
    assert by_name["D"]["y"] == pytest.approx(1000.0, abs=2e-3)

    assert res["statistics"]["degrees_of_freedom"] == 6
    assert res["statistics"]["converged"] is True
    # 无粗差时不应有超限标准化残差
    assert res["suspects"] == []
    # 平差后闭合差趋于 0
    hc = [c for c in res["loop_closures"] if c["type"] == "horizontal"]
    assert hc and all(c["post_misclosure_m"] < 1e-6 for c in hc)
    # 椭圆几何合理
    for s in res["stations"]:
        e = s["ellipse"]
        assert e["semi_major_m"] >= e["semi_minor_m"] > 0
        assert 0 <= e["major_azimuth_deg"] < 180


def test_gross_error_detected_and_pointed():
    payload = square_traverse_payload(gross_edge=("A", "B"), gross_kind="azimuth",
                                      gross_arcsec=30.0)
    req = AdjustmentRequest.model_validate(payload)
    res = run_adjustment(req)
    assert any(s["observation_id"] == "eAB" and s["component"] == "azimuth"
               for s in res["suspects"])
    o = next(o for o in res["observations"] if o["id"] == "eAB")
    assert o["suspect"] is True


def test_reversed_sight_hint():
    # 把 A->B 方位角写反 180°（前后视颠倒）
    payload = square_traverse_payload(
        gross_edge=("A", "B"), payload={"outlier_threshold": 2.0})
    for o in payload["observations"]:
        if o["id"] == "eAB":
            o["azimuth"] = o["azimuth"] + 180.0
    req = AdjustmentRequest.model_validate(payload)
    res = run_adjustment(req)
    comp = next(o for o in res["observations"] if o["id"] == "eAB")["components"]["azimuth"]
    assert comp["is_outlier"]
    assert "前后视写反" in comp["hint"]


def test_gon_unit_request():
    payload = square_traverse_payload()
    for o in payload["observations"]:
        o["azimuth"] = rad_to_deg(0)  # placeholder
    # 直接用 gon 重新构造
    P = {"A": (0.0, 0.0), "B": (1000.0, 0.0), "C": (1000.0, 1000.0), "D": (0.0, 1000.0)}
    obs = []
    for i, (f, t) in enumerate(
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")], start=1):
        dx, dy = P[t][0] - P[f][0], P[t][1] - P[f][1]
        obs.append({"from": f, "to": t,
                    "azimuth": math.degrees(math.atan2(dy, dx)) / 0.9,
                    "distance": math.hypot(dx, dy)})
    payload["observations"] = obs
    payload["units"] = {"angle": "gon"}
    req = AdjustmentRequest.model_validate(payload)
    res = run_adjustment(req)
    assert res["suspects"] == []


def test_leveling_network():
    req = AdjustmentRequest.model_validate(leveling_payload())
    res = run_adjustment(req)
    h = {s["name"]: s["h"] for s in res["stations"]}
    # 闭合差 +4 mm，按等权分配
    assert h["B2"] == pytest.approx(12.0, abs=0.01)
    assert h["B3"] == pytest.approx(11.0, abs=0.01)
    vc = [c for c in res["loop_closures"] if c["type"] == "height"]
    assert vc and abs(vc[0]["pre_misclosure_m"] - 0.004) < 1e-9
    assert abs(vc[0]["post_misclosure_m"]) < 1e-9


# ---------- 预检失败 ----------

def test_unknown_endpoint():
    payload = square_traverse_payload()
    payload["observations"][0]["to"] = "X9"
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(AdjustmentRequest.model_validate(payload))
    assert e.value.code == "unknown_endpoint"


def test_insufficient_datum():
    payload = square_traverse_payload()
    payload["known"] = [payload["known"][0]]
    payload["stations"] = ["B", "C", "D"]
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(AdjustmentRequest.model_validate(payload))
    assert e.value.code == "insufficient_datum"


def test_height_datum_required():
    payload = leveling_payload()
    payload["known"][0].pop("h")
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(AdjustmentRequest.model_validate(payload))
    assert e.value.code == "insufficient_height_datum"


def test_disconnected():
    payload = square_traverse_payload()
    # 切断与 D 相关的全部边
    payload["observations"] = [
        o for o in payload["observations"]
        if "D" not in (o["from"], o["to"])]
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(AdjustmentRequest.model_validate(payload))
    assert e.value.code in ("disconnected", "unobserved_station")


def test_rank_deficiency():
    # 2 个已知点 + 只有一条边确定的待求点，第二个待求点无观测 -> unobserved
    req = AdjustmentRequest(
        known=[KnownPoint(name="A", x=0, y=0), KnownPoint(name="B", x=1, y=0)],
        stations=["C", "D"],
        observations=[Observation(**{"from": "A", "to": "C",
                                     "azimuth": 45.0, "distance": 1.0})],
    )
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(req)
    assert e.value.code == "unobserved_station"


def test_self_loop_rejected():
    req = AdjustmentRequest(
        known=[KnownPoint(name="A", x=0, y=0), KnownPoint(name="B", x=1, y=0)],
        stations=["C"],
        observations=[
            Observation(**{"from": "A", "to": "C", "azimuth": 45.0, "distance": 1.0}),
            Observation(**{"from": "C", "to": "C", "distance": 1.0}),
        ],
    )
    with pytest.raises(AdjustmentError) as e:
        run_adjustment(req)
    assert e.value.code == "self_loop"


def test_duplicate_observation_warns_but_runs():
    payload = square_traverse_payload()
    dup = dict(payload["observations"][0])
    payload["observations"].append(dup)
    res = run_adjustment(AdjustmentRequest.model_validate(payload))
    assert res["duplicates"]
    assert any("重复观测" in w for w in res["warnings"])
