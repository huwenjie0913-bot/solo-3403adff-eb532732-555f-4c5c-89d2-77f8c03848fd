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
