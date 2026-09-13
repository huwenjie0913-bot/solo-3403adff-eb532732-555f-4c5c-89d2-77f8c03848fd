"""API 端到端：平差、错误码、存储、版本复算、预演。"""
from conftest import square_traverse_payload, leveling_payload


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_software_versions(client):
    r = client.get("/api/v1/software_versions")
    assert r.status_code == 200
    assert set(r.json()) >= {"python", "fastapi", "numpy", "scipy", "pydantic"}


def test_adjustment_saves_version(client):
    payload = square_traverse_payload()
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]["version_no"] == 1
    assert body["statistics"]["degrees_of_freedom"] == 6
    assert len(body["stations"]) == 2
    assert body["stations"][0]["ellipse"]["unit"] == "m"

    # 再次提交同名方案 -> version_no=2
    r2 = client.post("/api/v1/adjustments", json=payload)
    assert r2.json()["version"]["version_no"] == 2


def test_adjustment_validation_error(client):
    payload = square_traverse_payload()
    payload["known"] = [payload["known"][0]]
    payload["stations"] = ["B", "C", "D"]
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 400
    err = r.json()
    assert err["error"] == "insufficient_datum"


def test_unknown_endpoint_400(client):
    payload = square_traverse_payload()
    payload["observations"][0]["to"] = "GHOST"
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 400
    assert r.json()["error"] == "unknown_endpoint"


def test_version_trace_and_recompute(client):
    payload = square_traverse_payload(
        gross_edge=("A", "B"), gross_kind="azimuth", gross_arcsec=30.0)
    r = client.post("/api/v1/adjustments", json=payload)
    version_no = r.json()["version"]["version_no"]

    schemes = client.get("/api/v1/schemes").json()["schemes"]
    assert any(s["name"] == "square" for s in schemes)
    sid = next(s["id"] for s in schemes if s["name"] == "square")

    # 参数追溯
    v = client.get(f"/api/v1/schemes/{sid}/versions/{version_no}").json()
    assert v["request"]["outlier_threshold"] == 3.0
    assert v["software"]["numpy"]
    assert len(v["request"]["observations"]) == 5

    # 复算结果一致（默认不落库）
    rc = client.post(
        f"/api/v1/schemes/{sid}/versions/{version_no}/recompute").json()
    assert rc["recomputed_from"]["version_no"] == version_no
    assert rc["version"] is None
    assert any(s["observation_id"] == "eAB" for s in rc["suspects"])
    c_before = rc["stations"][0]["x"]

    versions = client.get(f"/api/v1/schemes/{sid}/versions").json()
    assert len(versions["versions"]) == 1  # 复算未新增版本

    # 显式 save=true 产生新版本
    rc2 = client.post(
        f"/api/v1/schemes/{sid}/versions/{version_no}/recompute?save=true").json()
    assert rc2["version"]["version_no"] == 2
    assert abs(rc2["stations"][0]["x"] - c_before) < 1e-9


def test_preview_disable_observation(client):
    payload = square_traverse_payload(
        gross_edge=("A", "B"), gross_kind="azimuth", gross_arcsec=30.0)
    r = client.post("/api/v1/preview", json={
        "base": payload,
        "disable": ["eAB"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    comp = body["comparison"]
    assert comp["disabled"] == ["eAB"]
    # 剔除后自由度减少 2（少了方位角+距离两行）
    assert comp["degrees_of_freedom"]["preview"] == \
        comp["degrees_of_freedom"]["base"] - 2
    # 剔除后不再有粗差
    assert body["preview"]["suspects"] == []
    # 点位移
    names = {s["name"]: s for s in comp["station_shifts"]}
    assert names["C"]["horizontal_shift_m"] >= 0.0


def test_preview_weight_override(client):
    payload = square_traverse_payload()
    r = client.post("/api/v1/preview", json={
        "base": payload,
        "weight_overrides": {"eAC": 10.0},
    })
    assert r.status_code == 200
    assert r.json()["comparison"]["sigma0_posterior"]["preview"] is not None


def test_preview_unknown_id_reported(client):
    payload = square_traverse_payload()
    r = client.post("/api/v1/preview", json={
        "base": payload, "disable": ["nope"]})
    assert r.status_code == 200
    assert "nope" in r.json()["comparison"]["unknown_observation_ids"]


def test_dms_string_angle(client):
    payload = square_traverse_payload()
    for o in payload["observations"]:
        import math
        deg = o["azimuth"]
        d = int(deg)
        mnt = int((deg - d) * 60)
        sec = ((deg - d) * 60 - mnt) * 60
        o["azimuth"] = f"{d}-{mnt:02d}-{sec:06.3f}"
    payload["units"] = {"angle": "dms"}
    r = client.post("/api/v1/adjustments", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["suspects"] == []


def test_leveling_endpoint_and_height_closure(client):
    r = client.post("/api/v1/adjustments", json=leveling_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    vc = [c for c in body["loop_closures"] if c["type"] == "height"]
    assert len(vc) == 1
    assert abs(vc[0]["pre_misclosure_m"] - 0.004) < 1e-9
    assert abs(vc[0]["post_misclosure_m"]) < 1e-9
    # 高程精度
    for s in body["stations"]:
        assert s["std_h_m"] > 0
