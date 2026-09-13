"""SQLite 持久化：方案、计算版本、软件版本。

每次平差保存一行版本，内容为完整请求 JSON（含单位、精度、阈值等全部参数），
因此可按编号无损复算并追溯所用参数。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from .models import AdjustmentRequest, DeformationRequest

DB_PATH = os.environ.get(
    "ADJUSTMENT_DB", os.path.join(os.path.dirname(__file__), "..", "data", "adjustment.db")
)

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS schemes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_id       INTEGER NOT NULL REFERENCES schemes(id),
    version_no      INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    summary_json    TEXT NOT NULL,
    software_json   TEXT NOT NULL,
    UNIQUE(scheme_id, version_no)
);

CREATE TABLE IF NOT EXISTS software_versions (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    versions    TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deformation_schemes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS deformation_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_id       INTEGER NOT NULL REFERENCES deformation_schemes(id),
    version_no      INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    summary_json    TEXT NOT NULL,
    software_json   TEXT NOT NULL,
    UNIQUE(scheme_id, version_no)
);
"""


def _connect() -> sqlite3.Connection:
    path = DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


_conn: Optional[sqlite3.Connection] = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _connect()
        _conn.executescript(SCHEMA)
    return _conn


def reset_for_tests(path: Optional[str] = None) -> None:
    """测试用：切换/清空数据库。"""
    global _conn, DB_PATH
    with _LOCK:
        if _conn is not None:
            _conn.close()
        if path is not None:
            DB_PATH = path
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        _conn = _connect()
        _conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------

def save_version(
    req: AdjustmentRequest,
    result: dict[str, Any],
    software: dict[str, str],
) -> dict[str, int]:
    """保存方案与新版本，返回 {scheme_id, version_id, version_no}。"""
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "statistics": result.get("statistics"),
        "suspects": result.get("suspects"),
        "warnings": result.get("warnings"),
        "stations": [
            {"name": s["name"], "x": s["x"], "y": s["y"], "h": s["h"]}
            for s in result.get("stations", [])
        ],
    }
    with _LOCK, conn:
        cur = conn.execute("SELECT id FROM schemes WHERE name IS ?", (req.name,))
        row = cur.fetchone()
        if row is None:
            cur = conn.execute("INSERT INTO schemes(name, created_at) VALUES (?, ?)",
                               (req.name, now))
            scheme_id = cur.lastrowid
            version_no = 1
        else:
            scheme_id = row["id"]
            cur = conn.execute(
                "SELECT COALESCE(MAX(version_no), 0) + 1 FROM versions WHERE scheme_id = ?",
                (scheme_id,))
            version_no = cur.fetchone()[0]

        cur = conn.execute(
            """INSERT INTO versions
               (scheme_id, version_no, created_at, request_json,
                summary_json, software_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (scheme_id, version_no, now,
             req.model_dump_json(by_alias=True),
             json.dumps(summary, ensure_ascii=False),
             json.dumps(software, ensure_ascii=False)),
        )
        return {
            "scheme_id": scheme_id,
            "version_id": cur.lastrowid,
            "version_no": version_no,
        }


def load_version(scheme_id: int, version_no: int) -> dict[str, Any]:
    conn = get_conn()
    row = conn.execute(
        """SELECT v.*, s.name AS scheme_name FROM versions v
           JOIN schemes s ON s.id = v.scheme_id
           WHERE v.scheme_id = ? AND v.version_no = ?""",
        (scheme_id, version_no),
    ).fetchone()
    if row is None:
        raise KeyError(f"未找到方案 {scheme_id} 的版本 {version_no}")
    return {
        "scheme_id": row["scheme_id"],
        "scheme_name": row["scheme_name"],
        "version_no": row["version_no"],
        "created_at": row["created_at"],
        "request": json.loads(row["request_json"]),
        "summary": json.loads(row["summary_json"]),
        "software": json.loads(row["software_json"]),
    }


def list_versions(scheme_id: Optional[int] = None) -> list[dict[str, Any]]:
    conn = get_conn()
    sql = """SELECT s.id AS scheme_id, s.name AS scheme_name,
                    v.version_no, v.id AS version_id, v.created_at, v.summary_json
             FROM versions v JOIN schemes s ON s.id = v.scheme_id"""
    params = ()
    if scheme_id is not None:
        sql += " WHERE s.id = ?"
        params = (scheme_id,)
    sql += " ORDER BY s.id, v.version_no"
    out = []
    for r in conn.execute(sql, params):
        item = dict(r)
        item["summary"] = json.loads(item.pop("summary_json"))
        out.append(item)
    return out


def list_schemes() -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT s.id, s.name, s.created_at, COUNT(v.id) AS version_count
           FROM schemes s LEFT JOIN versions v ON v.scheme_id = s.id
           GROUP BY s.id ORDER BY s.id"""
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 多期形变分析的方案与版本
# ---------------------------------------------------------------------------

def save_deformation_version(
    req: DeformationRequest,
    result: dict[str, Any],
    software: dict[str, str],
) -> dict[str, int]:
    """保存形变分析方案与新版本，返回 {scheme_id, version_id, version_no}。"""
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "reference_epoch": result.get("reference_epoch"),
        "parameters": result.get("parameters"),
        "summary": result.get("summary"),
        "epochs": [
            {k: e[k] for k in ("epoch", "time", "batch")}
            for e in result.get("epochs", [])
        ],
    }
    with _LOCK, conn:
        cur = conn.execute(
            "SELECT id FROM deformation_schemes WHERE name IS ?", (req.name,))
        row = cur.fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO deformation_schemes(name, created_at) VALUES (?, ?)",
                (req.name, now))
            scheme_id = cur.lastrowid
            version_no = 1
        else:
            scheme_id = row["id"]
            cur = conn.execute(
                "SELECT COALESCE(MAX(version_no), 0) + 1 "
                "FROM deformation_versions WHERE scheme_id = ?",
                (scheme_id,))
            version_no = cur.fetchone()[0]

        cur = conn.execute(
            """INSERT INTO deformation_versions
               (scheme_id, version_no, created_at, request_json,
                summary_json, software_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (scheme_id, version_no, now,
             req.model_dump_json(by_alias=True),
             json.dumps(summary, ensure_ascii=False),
             json.dumps(software, ensure_ascii=False)),
        )
        return {
            "scheme_id": scheme_id,
            "version_id": cur.lastrowid,
            "version_no": version_no,
        }


def load_deformation_version(scheme_id: int, version_no: int) -> dict[str, Any]:
    conn = get_conn()
    row = conn.execute(
        """SELECT v.*, s.name AS scheme_name FROM deformation_versions v
           JOIN deformation_schemes s ON s.id = v.scheme_id
           WHERE v.scheme_id = ? AND v.version_no = ?""",
        (scheme_id, version_no),
    ).fetchone()
    if row is None:
        raise KeyError(f"未找到形变分析方案 {scheme_id} 的版本 {version_no}")
    return {
        "scheme_id": row["scheme_id"],
        "scheme_name": row["scheme_name"],
        "version_no": row["version_no"],
        "created_at": row["created_at"],
        "request": json.loads(row["request_json"]),
        "summary": json.loads(row["summary_json"]),
        "software": json.loads(row["software_json"]),
    }


def list_deformation_versions(scheme_id: Optional[int] = None) -> list[dict[str, Any]]:
    conn = get_conn()
    sql = """SELECT s.id AS scheme_id, s.name AS scheme_name,
                    v.version_no, v.id AS version_id, v.created_at, v.summary_json
             FROM deformation_versions v
             JOIN deformation_schemes s ON s.id = v.scheme_id"""
    params = ()
    if scheme_id is not None:
        sql += " WHERE s.id = ?"
        params = (scheme_id,)
    sql += " ORDER BY s.id, v.version_no"
    out = []
    for r in conn.execute(sql, params):
        item = dict(r)
        item["summary"] = json.loads(item.pop("summary_json"))
        out.append(item)
    return out


def list_deformation_schemes() -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT s.id, s.name, s.created_at, COUNT(v.id) AS version_count
           FROM deformation_schemes s
           LEFT JOIN deformation_versions v ON v.scheme_id = s.id
           GROUP BY s.id ORDER BY s.id"""
    ).fetchall()
    return [dict(r) for r in rows]
