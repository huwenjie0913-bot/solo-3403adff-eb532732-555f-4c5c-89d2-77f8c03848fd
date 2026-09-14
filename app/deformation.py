"""多期形变分析。

同一控制网在不同日期复测时，单看两期坐标差会把仪器噪声误当成位移。
本模块的处理流程：

1. **各期解算**：每期观测分别做加权最小二乘平差（复用 adjustment 引擎），
   各期都以稳定基准点（共同控制点）为固定约束，从而把各期坐标对齐到同一
   稳定基准；
2. **协方差传播**：位移 d = X_k − X_0 的协方差 Σ_d = Σ_0 + Σ_k
   （各期观测相互独立，参考期误差一并计入）；
3. **位移与检验**：对每个站点计算三维位移、χ² 缩放的置信椭球，以及
   联合显著性检验 T = dᵀΣ_d⁻¹d ~ χ²(dim)（同时给出各分量的 z 检验）；
4. **速度与超阈值时刻**：速度 = 位移 / 历元间隔（年），并按时间顺序
   找出首个位移模长超过阈值的期次；
5. **基准比较**：可指定替代基准点重算全部期次，用 Helmert 相似变换把
   替代基准下的各期坐标对齐回主基准（协方差经 Jacobian 传播），比较两种
   基准方案下各点“是否显著 / 是否超阈值”的结论差异。

预检错误（均返回 400 并定位问题）：
* ``insufficient_datum_points``  基准点不足（<2 个共同控制点）
* ``datum_point_not_observed``   基准点未在每期观测中出现
* ``epoch_time_order``           期次时间倒序或相同
* ``epoch_point_mismatch``       历期间点名不一致（逐期列出缺失点）
* ``epoch_selection_invalid``    两期选择不合法
* ``alternative_datum_invalid``  替代基准不合法
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

import numpy as np
from scipy import stats

from .adjustment import AdjustmentError, run_adjustment
from .models import AdjustmentRequest, DeformationRequest, KnownPoint

DAYS_PER_YEAR = 365.25
SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# 1. 预检
# ---------------------------------------------------------------------------

def _parse_time(raw: str, epoch_key: str) -> datetime:
    try:
        return datetime.fromisoformat(str(raw).strip())
    except ValueError:
        raise AdjustmentError(
            f"期次 {epoch_key} 的时间 {raw!r} 无法解析"
            "（需要 ISO 8601，如 2026-03-01 或 2026-03-01T08:30:00）",
            "bad_epoch_time", {"epoch": epoch_key, "time": raw},
        )


def _epoch_point_names(epoch) -> set[str]:
    names: set[str] = set()
    for o in epoch.observations:
        names.add(o.frm)
        names.add(o.to)
    for b in getattr(epoch, "baselines", []):
        names.add(b.frm)
        names.add(b.to)
    return names


def _validate(req: DeformationRequest):
    """返回 (期次->时刻, 全部共同点(排序), 监测点列表)。"""
    if len(req.epochs) < 2:
        raise AdjustmentError(
            f"多期形变分析至少需要 2 期观测（当前 {len(req.epochs)} 期）",
            "epoch_count", {"epoch_count": len(req.epochs)},
        )
    keys = [e.epoch for e in req.epochs]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise AdjustmentError(
            "期次标识重复: " + ", ".join(dup), "duplicate_epoch", {"epochs": dup}
        )

    times = {e.epoch: _parse_time(e.time, e.epoch) for e in req.epochs}
    aware = [t.tzinfo is not None and t.tzinfo.utcoffset(None) is not None
             for t in times.values()]
    if any(aware) and not all(aware):
        raise AdjustmentError(
            "各期时间不能混用带时区与不带时区的写法", "bad_epoch_time",
            {"times": {k: t.isoformat() for k, t in times.items()}},
        )
    for a, b in zip(req.epochs, req.epochs[1:]):
        if times[b.epoch] <= times[a.epoch]:
            raise AdjustmentError(
                f"期次时间必须严格递增：{a.epoch}（{a.time}）之后不能是 "
                f"{b.epoch}（{b.time}）——时间倒序或相同",
                "epoch_time_order",
                {
                    "previous": {"epoch": a.epoch, "time": a.time},
                    "next": {"epoch": b.epoch, "time": b.time},
                },
            )

    # 历期间点名一致性：每期观测到的点集必须完全相同
    point_sets = {e.epoch: _epoch_point_names(e) for e in req.epochs}
    union: set[str] = set().union(*point_sets.values())
    missing = {k: sorted(union - s) for k, s in point_sets.items() if union - s}
    if missing:
        raise AdjustmentError(
            "历期间点名不一致："
            + "；".join(f"期次 {k} 缺少 {', '.join(v)}" for k, v in missing.items()),
            "epoch_point_mismatch",
            {"all_points": sorted(union), "missing_by_epoch": missing},
        )
    common = sorted(union)

    datum_names = [p.name for p in req.datum_points]
    if len(set(datum_names)) != len(datum_names):
        dup = sorted({n for n in datum_names if datum_names.count(n) > 1})
        raise AdjustmentError("基准点重名: " + ", ".join(dup),
                              "duplicate_point", {"points": dup})
    if len(datum_names) < 2:
        raise AdjustmentError(
            f"基准点不足：多期形变分析至少需要 2 个共同控制点作为稳定基准"
            f"（当前 {len(datum_names)} 个）",
            "insufficient_datum_points", {"datum_count": len(datum_names)},
        )
    not_observed = [n for n in datum_names if n not in common]
    if not_observed:
        raise AdjustmentError(
            "基准点未在每期观测中出现（共同控制点必须每期都观测）: "
            + ", ".join(not_observed),
            "datum_point_not_observed", {"points": not_observed},
        )

    if req.monitor_points:
        unknown = [m for m in req.monitor_points if m not in common]
        if unknown:
            raise AdjustmentError(
                "监测点未在观测中出现: " + ", ".join(unknown),
                "unknown_monitor_point", {"points": unknown},
            )
        overlap = sorted(set(req.monitor_points) & set(datum_names))
        if overlap:
            raise AdjustmentError(
                "监测点与基准点重名: " + ", ".join(overlap),
                "duplicate_point", {"points": overlap},
            )
        monitors = list(req.monitor_points)
    else:
        monitors = sorted(set(common) - set(datum_names))
    if not monitors:
        raise AdjustmentError(
            "没有监测点：除基准点外各期至少还要有一个共同测站点",
            "no_monitor_points",
        )

    if req.epoch_selection is not None:
        sel = req.epoch_selection
        if len(sel) != 2 or len(set(sel)) != 2:
            raise AdjustmentError(
                f"epoch_selection 用于两期重算，必须恰好给出 2 个不同的期次标识"
                f"（当前 {len(sel)} 个）；缺省则使用全部期次连续分析",
                "epoch_selection_invalid", {"selection": sel},
            )
        unknown_sel = [k for k in sel if k not in times]
        if unknown_sel:
            raise AdjustmentError(
                "epoch_selection 中存在未知期次: " + ", ".join(unknown_sel),
                "epoch_selection_invalid", {"unknown": unknown_sel},
            )

    if req.alternative_datum is not None:
        alt = req.alternative_datum
        if len(alt) < 2 or len(set(alt)) != len(alt):
            raise AdjustmentError(
                "替代基准至少需要 2 个互不相同的点",
                "alternative_datum_invalid", {"alternative_datum": alt},
            )
        unknown_alt = [n for n in alt if n not in common]
        if unknown_alt:
            raise AdjustmentError(
                "替代基准点未在观测中出现: " + ", ".join(unknown_alt),
                "alternative_datum_invalid", {"unknown": unknown_alt},
            )
    return times, common, monitors


# ---------------------------------------------------------------------------
# 2. 各期解算（复用平差引擎，基准点固定 = 对齐到稳定基准）
# ---------------------------------------------------------------------------

def _solve_epoch(req: DeformationRequest, epoch,
                 known: list[KnownPoint], stations: list[str]) -> dict[str, Any]:
    adj = AdjustmentRequest(
        name=None,
        known=known,
        stations=stations,
        observations=epoch.observations,
        baselines=getattr(epoch, "baselines", []),
        units=req.units,
        accuracy=req.accuracy,
        outlier_threshold=req.outlier_threshold,
        confidence=req.confidence,
        save=False,
    )
    res = run_adjustment(adj)
    sol = res["_internal"]["sol"]
    sigma0 = res["statistics"]["sigma0_posterior"]
    cov = (sigma0 ** 2) * np.linalg.inv(sol["N"])
    pidx = sol["pidx"]

    # 每个点的状态：三维坐标向量 + 3x3 协方差（基准点协方差为 0）
    state: dict[str, dict[str, Any]] = {}
    for p in known:
        state[p.name] = {
            "vec": np.array([p.x, p.y, p.h if p.h is not None else math.nan]),
            "cov": np.zeros((3, 3)),
            "has_h": p.h is not None,
        }
    pos = {"x": 0, "y": 1, "h": 2}
    for st in stations:
        comps = pidx.get(st, {})
        has_h = "h" in comps
        vec = np.array([
            sol["coords"][st][0],
            sol["coords"][st][1],
            sol["heights"][st] if has_h else math.nan,
        ])
        m = np.zeros((3, 3))
        for c1, i in pos.items():
            if c1 not in comps:
                continue
            for c2, j in pos.items():
                if c2 not in comps:
                    continue
                m[i, j] = cov[comps[c1], comps[c2]]
        state[st] = {"vec": vec, "cov": m, "has_h": has_h}

    obs_ids: list[str] = []
    ids_by_point: dict[str, list[str]] = {}
    for i, o in enumerate(epoch.observations, start=1):
        oid = o.id or f"o{i}"
        obs_ids.append(oid)
        ids_by_point.setdefault(o.frm, []).append(oid)
        ids_by_point.setdefault(o.to, []).append(oid)
    for i, b in enumerate(getattr(epoch, "baselines", []), start=1):
        bid = b.id or f"b{i}"
        obs_ids.append(bid)
        ids_by_point.setdefault(b.frm, []).append(bid)
        ids_by_point.setdefault(b.to, []).append(bid)
    return {
        "state": state, "result": res,
        "obs_ids": obs_ids, "ids_by_point": ids_by_point,
    }


# ---------------------------------------------------------------------------
# 3. Helmert 相似变换（替代基准 -> 主基准对齐，协方差经 Jacobian 传播）
# ---------------------------------------------------------------------------

def _helmert_2d(src: dict[str, np.ndarray], dst: dict[str, np.ndarray],
                names: list[str]) -> np.ndarray:
    """估计四参数相似变换 dst ≈ T(src)：x' = tx + a·x − b·y；y' = ty + b·x + a·y。"""
    rows, rhs = [], []
    for n in names:
        x, y = float(src[n][0]), float(src[n][1])
        X, Y = float(dst[n][0]), float(dst[n][1])
        rows.append([1.0, 0.0, x, -y])
        rhs.append(X)
        rows.append([0.0, 1.0, y, x])
        rhs.append(Y)
    params, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return params  # tx, ty, a, b


def _apply_helmert(state: dict[str, dict[str, Any]], params: np.ndarray,
                   h_shift: float) -> dict[str, dict[str, Any]]:
    tx, ty, a, b = (float(v) for v in params)
    j2 = np.array([[a, -b], [b, a]])
    out: dict[str, dict[str, Any]] = {}
    for name, st in state.items():
        xy = np.array([tx, ty]) + j2 @ st["vec"][:2]
        h = st["vec"][2] + h_shift if st["has_h"] else st["vec"][2]
        j = np.eye(3)
        j[:2, :2] = j2
        out[name] = {
            "vec": np.array([xy[0], xy[1], h]),
            "cov": j @ st["cov"] @ j.T,
            "has_h": st["has_h"],
        }
    return out


# ---------------------------------------------------------------------------
# 4. 位移、置信椭球、显著性检验
# ---------------------------------------------------------------------------

def _ellipsoid(sv: np.ndarray, dim: int, confidence: float) -> dict[str, Any]:
    """由位移协方差求置信椭球（dim=3）或椭圆（dim=2）。"""
    k = float(stats.chi2.ppf(confidence, dim))
    w, v = np.linalg.eigh(sv)
    w = np.clip(w, 0.0, None)
    order = np.argsort(w)[::-1]
    axes = [float(x) for x in np.sqrt(w[order] * k)]
    major = v[:, order[0]]
    out: dict[str, Any] = {
        "dimension": dim,
        "confidence": confidence,
        "chi2_scale": k,
        "semi_axes_m": axes,
        "major_axis_azimuth_deg": float(
            math.degrees(math.atan2(float(major[1]), float(major[0]))) % 360.0
        ),
    }
    if dim == 3:
        out["major_axis_dip_deg"] = float(
            math.degrees(math.asin(float(np.clip(major[2], -1.0, 1.0))))
        )
    return out


def _analyze(req: DeformationRequest, metas: list[dict[str, Any]],
             states: list[dict[str, dict[str, Any]]],
             solved: list[dict[str, Any]],
             monitors: list[str], datum_names: list[str]) -> list[dict[str, Any]]:
    """对选中期次逐点计算位移、速度、置信椭球与联合显著性检验。"""
    threshold = req.displacement_threshold
    alpha = 1.0 - req.confidence
    ref = states[0]
    points_out: list[dict[str, Any]] = []
    for name in list(monitors) + list(datum_names):
        st0 = ref[name]
        role = "datum" if name in datum_names else "monitor"
        disps: list[dict[str, Any]] = []
        significant = False
        min_p: Optional[float] = None
        max_mag = 0.0
        first_exceed: Optional[dict[str, str]] = None
        for k in range(1, len(states)):
            stk = states[k][name]
            has_h = bool(st0["has_h"] and stk["has_h"])
            dim = 3 if has_h else 2
            dv = (stk["vec"] - st0["vec"])[:dim]
            sv = (stk["cov"] + st0["cov"])[:dim, :dim]
            d2d = float(math.hypot(float(dv[0]), float(dv[1])))
            d3d = float(np.linalg.norm(dv)) if dim == 3 else None
            mag = d3d if d3d is not None else d2d
            exceeds = bool(mag >= threshold)
            if exceeds and first_exceed is None:
                first_exceed = {"epoch": metas[k]["epoch"], "time": metas[k]["time"]}
            max_mag = max(max_mag, mag)

            test = None
            ell = None
            if np.any(sv):  # 基准点协方差恒为 0，不做检验
                t_stat = max(float(dv @ np.linalg.solve(sv, dv)), 0.0)
                p_val = float(1.0 - stats.chi2.cdf(t_stat, dim))
                comps = []
                for i, cname in enumerate(("x", "y", "h")[:dim]):
                    sd = float(math.sqrt(max(sv[i, i], 0.0)))
                    z = float(dv[i] / sd) if sd > 0 else None
                    comps.append({
                        "component": cname,
                        "difference_m": float(dv[i]),
                        "std_m": sd,
                        "z": z,
                        "p_value": (
                            float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
                            if z is not None else None
                        ),
                    })
                test = {
                    "statistic": t_stat,
                    "df": dim,
                    "p_value": p_val,
                    "significant": bool(p_val < alpha),
                    "components": comps,
                }
                significant = significant or test["significant"]
                min_p = p_val if min_p is None else min(min_p, p_val)
                ell = _ellipsoid(sv, dim, req.confidence)

            years = metas[k]["days_from_reference"] / DAYS_PER_YEAR
            velocity = None
            if years > 0:
                velocity = {
                    "vx_m_per_year": float(dv[0] / years),
                    "vy_m_per_year": float(dv[1] / years),
                    "vh_m_per_year": float(dv[2] / years) if dim == 3 else None,
                    "speed_m_per_year": float(mag / years),
                }
            disps.append({
                "epoch": metas[k]["epoch"],
                "time": metas[k]["time"],
                "batch": metas[k]["batch"],
                "days_from_reference": metas[k]["days_from_reference"],
                "dx_m": float(dv[0]),
                "dy_m": float(dv[1]),
                "dh_m": float(dv[2]) if dim == 3 else None,
                "d2d_m": d2d,
                "d3d_m": d3d,
                "exceeds_threshold": exceeds,
                "velocity": velocity,
                "covariance_m2": [[float(x) for x in row] for row in sv],
                "ellipsoid": ell,
                "test": test,
                "observation_ids": solved[k]["ids_by_point"].get(name, []),
            })
        points_out.append({
            "name": name,
            "role": role,
            "has_height": bool(st0["has_h"]),
            "significant": significant,
            "min_p_value": min_p,
            "max_displacement_m": max_mag,
            "first_exceed_epoch": first_exceed["epoch"] if first_exceed else None,
            "first_exceed_time": first_exceed["time"] if first_exceed else None,
            "displacements": disps,
        })
    return points_out


# ---------------------------------------------------------------------------
# 5. 替代基准：重算 + 对齐回主基准 + 结论比较
# ---------------------------------------------------------------------------

def _alternative_datum_analysis(req: DeformationRequest,
                                sel_epochs, metas, common, datum_names,
                                ref_state, primary_points) -> dict[str, Any]:
    alt_names = list(req.alternative_datum)
    # 替代基准点的参考坐标取主基准下参考期的平差值
    alt_known = []
    for n in alt_names:
        st = ref_state[n]
        alt_known.append(KnownPoint(
            name=n,
            x=float(st["vec"][0]),
            y=float(st["vec"][1]),
            h=float(st["vec"][2]) if st["has_h"] else None,
        ))
    alt_stations = sorted(set(common) - set(alt_names))
    solved = [_solve_epoch(req, e, alt_known, alt_stations) for e in sel_epochs]

    # 以主基准点为公共点，把各期对齐回主基准（Helmert 变换 + 协方差传播）
    dst = {p.name: np.array([p.x, p.y]) for p in req.datum_points}
    h_ref = {p.name: p.h for p in req.datum_points}
    aligned: list[dict[str, dict[str, Any]]] = []
    alignment_params: list[dict[str, Any]] = []
    for meta, s in zip(metas, solved):
        src = {n: s["state"][n]["vec"] for n in datum_names}
        params = _helmert_2d(src, dst, datum_names)
        shifts = [
            h_ref[n] - float(s["state"][n]["vec"][2])
            for n in datum_names
            if h_ref[n] is not None and s["state"][n]["has_h"]
        ]
        h_shift = float(np.mean(shifts)) if shifts else 0.0
        aligned.append(_apply_helmert(s["state"], params, h_shift))
        tx, ty, a, b = (float(v) for v in params)
        alignment_params.append({
            "epoch": meta["epoch"],
            "tx_m": tx, "ty_m": ty,
            "rotation_arcsec": math.degrees(math.atan2(b, a)) * 3600.0,
            "scale": math.hypot(a, b),
            "h_shift_m": h_shift,
        })

    alt_monitors = sorted(set(common) - set(alt_names))
    alt_points = _analyze(req, metas, aligned, solved, alt_monitors, alt_names)

    pa = {p["name"]: p for p in primary_points}
    aa = {p["name"]: p for p in alt_points}

    def agg(p):
        if p is None:
            return None
        return {
            "role": p["role"],
            "significant": p["significant"],
            "max_displacement_m": p["max_displacement_m"],
            "min_p_value": p["min_p_value"],
            "first_exceed_epoch": p["first_exceed_epoch"],
        }

    rows, changed = [], []
    for name in sorted(set(pa) | set(aa)):
        P, A = pa.get(name), aa.get(name)
        sig_diff = bool(P and P["significant"]) != bool(A and A["significant"])
        exc_diff = (bool(P and P["first_exceed_epoch"] is not None)
                    != bool(A and A["first_exceed_epoch"] is not None))
        rows.append({
            "name": name,
            "primary": agg(P),
            "alternative": agg(A),
            "conclusion_changed": bool(sig_diff or exc_diff),
        })
        if sig_diff or exc_diff:
            changed.append(name)
    return {
        "primary_datum": list(datum_names),
        "alternative_datum": alt_names,
        "alignment_to_primary": alignment_params,
        "points": rows,
        "conclusion_changed_points": changed,
    }


# ---------------------------------------------------------------------------
# 6. 主入口
# ---------------------------------------------------------------------------

def run_deformation(req: DeformationRequest) -> dict[str, Any]:
    """执行多期形变分析，返回结果字典。"""
    times, common, monitors = _validate(req)
    datum_names = [p.name for p in req.datum_points]
    epoch_by_key = {e.epoch: e for e in req.epochs}
    if req.epoch_selection is not None:
        sel_keys = sorted(req.epoch_selection, key=lambda k: times[k])
    else:
        sel_keys = [e.epoch for e in req.epochs]
    sel_epochs = [epoch_by_key[k] for k in sel_keys]

    t0 = times[sel_keys[0]]
    metas = []
    for e in sel_epochs:
        dt = times[e.epoch]
        metas.append({
            "epoch": e.epoch,
            "time": dt.isoformat(),
            "batch": e.batch,
            "is_reference": e.epoch == sel_keys[0],
            "days_from_reference": (dt - t0).total_seconds() / SECONDS_PER_DAY,
        })

    solved = [_solve_epoch(req, e, req.datum_points, monitors) for e in sel_epochs]
    states = [s["state"] for s in solved]
    points = _analyze(req, metas, states, solved, monitors, datum_names)

    epochs_out = []
    for meta, s in zip(metas, solved):
        res = s["result"]
        epochs_out.append({
            **meta,
            "observation_ids": s["obs_ids"],
            "statistics": res["statistics"],
            "suspects": res["suspects"],
            "warnings": res["warnings"],
            "coordinates": {
                n: {
                    "x": float(st["vec"][0]),
                    "y": float(st["vec"][1]),
                    "h": float(st["vec"][2]) if st["has_h"] else None,
                }
                for n, st in s["state"].items()
            },
        })

    comparison = None
    if req.alternative_datum:
        comparison = _alternative_datum_analysis(
            req, sel_epochs, metas, common, datum_names, states[0], points)

    exceeding = [p["name"] for p in points if p["first_exceed_epoch"] is not None]
    significant = [p["name"] for p in points if p["significant"]]
    return {
        "status": "ok",
        "name": req.name,
        "parameters": {
            "datum_points": datum_names,
            "alternative_datum": req.alternative_datum,
            "monitor_points": monitors,
            "displacement_threshold": req.displacement_threshold,
            "confidence": req.confidence,
            "epoch_selection": sel_keys,
            "outlier_threshold": req.outlier_threshold,
            "units": req.units.model_dump(),
            "accuracy": req.accuracy.model_dump(),
        },
        "reference_epoch": sel_keys[0],
        "epochs": epochs_out,
        "points": points,
        "summary": {
            "points_total": len(points),
            "monitor_points": len(monitors),
            "epochs_analyzed": len(sel_keys),
            "points_exceeding_threshold": exceeding,
            "points_significant": significant,
            "max_displacement_m": max(
                (p["max_displacement_m"] for p in points), default=0.0),
        },
        "datum_comparison": comparison,
        "warnings": [],
    }
