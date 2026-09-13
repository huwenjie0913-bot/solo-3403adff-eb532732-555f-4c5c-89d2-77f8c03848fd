"""测试夹具与样例网络。"""
import math
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    monkeypatch.setenv("ADJUSTMENT_DB", db)
    from app import storage
    storage.reset_for_tests(db)
    from app.main import app
    return TestClient(app)


def square_traverse_payload(**overrides):
    """闭合导线 A-B-C-D-A + 对角线 A-C，A/B 已知，C/D 待求。"""
    P = {"A": (0.0, 0.0), "B": (1000.0, 0.0),
         "C": (1000.0, 1000.0), "D": (0.0, 1000.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]
    obs = []
    for i, (f, t) in enumerate(edges, start=1):
        dx = P[t][0] - P[f][0]
        dy = P[t][1] - P[f][1]
        az = math.degrees(math.atan2(dy, dx))
        if overrides.get("gross_edge") == (f, t):
            kind = overrides.get("gross_kind", "azimuth")
            if kind == "azimuth":
                az += overrides.get("gross_arcsec", 20.0) / 3600.0
        obs.append({
            "id": f"e{f}{t}",
            "from": f, "to": t,
            "azimuth": round(az, 8),
            "distance": math.hypot(dx, dy),
        })
    payload = {
        "name": "square",
        "known": [
            {"name": "A", "x": 0.0, "y": 0.0},
            {"name": "B", "x": 1000.0, "y": 0.0},
        ],
        "stations": ["C", "D"],
        "observations": obs,
    }
    payload.update(overrides.get("payload", {}))
    return payload


def leveling_payload():
    """水准环 B1-B2-B3-B1，B1 已知高程 10 m。"""
    # 真值高程 10, 12, 11
    obs = [
        {"id": "l1", "from": "B1", "to": "B2", "dh": 2.001},
        {"id": "l2", "from": "B2", "to": "B3", "dh": -1.002},
        {"id": "l3", "from": "B3", "to": "B1", "dh": -0.995},
    ]
    return {
        "name": "level",
        "known": [{"name": "B1", "x": 0, "y": 0, "h": 10.0}],
        "stations": ["B2", "B3"],
        "observations": obs,
    }


def deformation_payload(**overrides):
    """三期形变监测网：D1/D2 为稳定基准点，M1/M2 为监测点。

    E3 期 M1 发生位移 (+5cm, +2cm, +1cm)，M2 保持不动；观测值由真值
    加确定性微小噪声（±2mm / ±1″ / ±1mm）生成。
    """
    datum = {"D1": (0.0, 0.0, 100.0), "D2": (1000.0, 0.0, 100.0)}
    truth = {
        "E1": {"M1": (500.0, 400.0, 101.0), "M2": (500.0, 600.0, 102.0)},
        "E2": {"M1": (500.0, 400.0, 101.0), "M2": (500.0, 600.0, 102.0)},
        "E3": {"M1": (500.05, 400.02, 101.01), "M2": (500.0, 600.0, 102.0)},
    }
    times = {"E1": "2026-01-01", "E2": "2026-03-01", "E3": "2026-05-01"}
    # D2 只位于完整边的 from 端（D2->M1、D2->M2），替代基准中转为待求点时
    # 初值需沿观测几何反向传播才能得到（回归场景）
    edges = [("D1", "M1"), ("D2", "M1"), ("D1", "M2"), ("D2", "M2"), ("M1", "M2")]
    dh_edges = [("D1", "M1"), ("D1", "M2"), ("M1", "M2")]

    def noise(i, step):
        return ((i % 3) - 1) * step

    epochs = []
    for ek in ("E1", "E2", "E3"):
        pts = {**datum, **truth[ek]}
        obs = []
        i = 0
        for f, t in edges:
            i += 1
            dx = pts[t][0] - pts[f][0]
            dy = pts[t][1] - pts[f][1]
            az = math.degrees(math.atan2(dy, dx)) + noise(i, 1.0 / 3600.0)
            dist = math.hypot(dx, dy) + noise(i, 0.002)
            obs.append({
                "id": f"{ek}-{f}{t}", "from": f, "to": t,
                "azimuth": round(az, 8), "distance": round(dist, 4),
            })
        for f, t in dh_edges:
            i += 1
            dh = pts[t][2] - pts[f][2] + noise(i, 0.001)
            obs.append({
                "id": f"{ek}-h{f}{t}", "from": f, "to": t, "dh": round(dh, 4),
            })
        epochs.append({
            "epoch": ek, "time": times[ek], "batch": f"batch-{ek}",
            "observations": obs,
        })
    payload = {
        "name": "dam-monitor",
        "datum_points": [
            {"name": "D1", "x": 0.0, "y": 0.0, "h": 100.0},
            {"name": "D2", "x": 1000.0, "y": 0.0, "h": 100.0},
        ],
        "epochs": epochs,
        "displacement_threshold": 0.02,
    }
    payload.update(overrides.get("payload", {}))
    return payload
