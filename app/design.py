"""测前网形设计（first-order design + 可靠性设计）核心。

外业前只知道拟设点位的近似坐标、候选观测及其先验精度，本模块在**不进行外业
观测**的前提下：

1. 由候选方向角/平距/高差/GNSS 基线在近似坐标处线性化，组建（白化后的）
   设计矩阵 A、权阵 W 与法方程 N = AᵀWA，检查基准、连通性与秩；
2. 先验协方差取 Σx = N⁻¹（σ0=1），给出点位中误差、置信误差椭圆与三维椭球；
3. 逐观测计算冗余度（多余观测分量）r_i = 1 − h_i，以及 Baarda 数据探测意义下的
   **最小可探测粗差 MDB**（标量 λ0 = z(1−α/2)+z(1−β)；基线按白化空间 χ²(3)
   备择检验反算非中心参数）与其**坐标影响 MDE/BNR**（∇x = N⁻¹AᵀW·∇l）；
4. 以“目标违反度”为边际精度收益，贪心挑选候选观测：先补结构（连通/秩），
   再按收益/成本逐次加入，直到精度与可靠性目标全部满足、预算用尽或无收益；
5. 支持锁定（必选/剔除）与预算约束，并给出两套网形的成本、最弱点精度、
   可靠性与秩对比。

内部单位与平差模块一致：弧度、米；GNSS 基线经 Cholesky 白化。
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from scipy import stats

from . import __version__
from .adjustment import (
    AdjustmentError,
    _baseline_model,
    _model_and_jacobian,
    _normalized_baselines,
    _normalized_observations,
    _rank_deficiency_details,
    _validate_network,
)
from .models import (
    AccuracyGoals,
    Baseline,
    DesignReplayOptions,
    NetworkDesignRequest,
    Observation,
)
from .units import rad_to_deg

DESIGN_ALGORITHM_VERSION = "nd-1.0.0"
"""网形设计算法版本（随方案写入 SQLite，复演时回显）。"""

_R_FLOOR = 1e-12
_GAIN_EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. 预处理：复用平差模块的归一化与预检
# ---------------------------------------------------------------------------

def _build_context(req: NetworkDesignRequest):
    """把设计请求归一化为内部候选池（复用平差模块的单位/协方差/基准检查）。"""
    known: dict[str, Any] = {}
    for p in req.known:
        if p.name in known:
            raise AdjustmentError(f"已知点重名: {p.name}", "duplicate_point")
        known[p.name] = p
    approx: dict[str, Any] = {}
    for p in req.points:
        if p.name in known:
            raise AdjustmentError(f"点 {p.name} 同时出现在已知点与待定点中",
                                  "duplicate_point")
        if p.name in approx:
            raise AdjustmentError(f"待定点重名: {p.name}", "duplicate_point")
        approx[p.name] = p

    # 解析候选 id（与平差模块缺省规则一致）并检查冲突
    obs_ids, bl_ids = [], []
    for i, o in enumerate(req.observations, start=1):
        obs_ids.append(o.id or f"o{i}")
    for i, b in enumerate(req.baselines, start=1):
        bl_ids.append(b.id or f"b{i}")
    if len(set(obs_ids)) != len(obs_ids):
        dup = sorted({x for x in obs_ids if obs_ids.count(x) > 1})
        raise AdjustmentError("候选观测编号重复: " + ", ".join(dup),
                              "duplicate_point", {"ids": dup})
    if len(set(bl_ids)) != len(bl_ids):
        dup = sorted({x for x in bl_ids if bl_ids.count(x) > 1})
        raise AdjustmentError("候选基线编号重复: " + ", ".join(dup),
                              "duplicate_point", {"ids": dup})
    clash = sorted(set(obs_ids) & set(bl_ids))
    if clash:
        raise AdjustmentError("候选观测与基线编号冲突: " + ", ".join(clash),
                              "duplicate_point", {"ids": clash})
    all_ids = set(obs_ids) | set(bl_ids)

    def check_locks(ids: list[str], code: str, label: str):
        unknown = sorted(set(ids) - all_ids)
        if unknown:
            raise AdjustmentError(
                f"{label}引用了不存在的候选 id: " + ", ".join(unknown),
                code, {"ids": unknown})

    check_locks(req.lock_included, "unknown_lock_id", "lock_included")
    check_locks(req.lock_excluded, "unknown_lock_id", "lock_excluded")
    for o in req.observations:
        if o.required and o.excluded:
            raise AdjustmentError(
                f"观测 {o.id} 同时标记为必选与强制不选", "conflicting_lock")
    for b in req.baselines:
        if b.required and b.excluded:
            raise AdjustmentError(
                f"基线 {b.id} 同时标记为必选与强制不选", "conflicting_lock")

    def coord_of(name):
        if name in known:
            return known[name].x, known[name].y, known[name].h
        p = approx[name]
        return p.x, p.y, p.h

    # 构造平差模块的模型对象（基线缺省分量由近似坐标反算）
    obs_models: list[Observation] = []
    meta: dict[str, dict[str, Any]] = {}
    for i, o in enumerate(req.observations, start=1):
        oid = obs_ids[i - 1]
        for end in (o.frm, o.to):
            if end not in known and end not in approx:
                raise AdjustmentError(
                    f"观测 {oid} 的端点 {end} 既不是已知点也不是待定点",
                    "unknown_endpoint", {"id": oid, "endpoint": end})
        if o.frm == o.to:
            raise AdjustmentError(
                f"观测 {oid} 起终点相同，不允许自环边", "self_loop", {"id": oid})
        if o.azimuth is None and o.distance is None and o.dh is None:
            raise AdjustmentError(
                f"观测 {oid} 未给任何候选分量（azimuth/distance/dh 至少一项）",
                "empty_observation", {"id": oid})
        # 高差边两端若为无高程的已知点，后续无法建立方程
        if o.dh is not None:
            for end in (o.frm, o.to):
                if end in known and known[end].h is None:
                    raise AdjustmentError(
                        f"高差观测 {oid} 的端点 {end} 是无已知高程的已知点",
                        "height_endpoint_without_height",
                        {"id": oid, "endpoint": end})
        obs_models.append(Observation(
            id=oid, frm=o.frm, to=o.to,
            azimuth=o.azimuth, distance=o.distance, dh=o.dh,
            std_azimuth=o.std_azimuth, std_distance=o.std_distance,
            std_dh=o.std_dh, weight=1.0,
        ))
        meta[oid] = {"cost": float(o.cost), "required": bool(o.required),
                     "excluded": bool(o.excluded), "kind": "observation",
                     "frm": o.frm, "to": o.to}

    bl_models: list[Baseline] = []
    for i, b in enumerate(req.baselines, start=1):
        bid = bl_ids[i - 1]
        for end in (b.frm, b.to):
            if end not in known and end not in approx:
                raise AdjustmentError(
                    f"基线 {bid} 的端点 {end} 既不是已知点也不是待定点",
                    "unknown_endpoint", {"id": bid, "endpoint": end})
        if b.frm == b.to:
            raise AdjustmentError(
                f"基线 {bid} 起终点相同，不允许自环边", "self_loop", {"id": bid})
        xf, yf, hf = coord_of(b.frm)
        xt, yt, ht = coord_of(b.to)
        dx = b.dx if b.dx is not None else xt - xf
        dy = b.dy if b.dy is not None else yt - yf
        if b.dh is not None:
            dh = b.dh
        else:
            if hf is None or ht is None:
                raise AdjustmentError(
                    f"基线 {bid} 未给 dh 且端点缺少近似高程，无法反算三维向量",
                    "no_initial_coordinates", {"id": bid})
            dh = ht - hf
        bl_models.append(Baseline(
            id=bid, **{"from": b.frm}, to=b.to,
            dx=dx, dy=dy, dh=dh, covariance=b.covariance,
            unit=b.unit, weight=b.weight,
        ))
        meta[bid] = {"cost": float(b.cost), "required": bool(b.required),
                     "excluded": bool(b.excluded), "kind": "baseline",
                     "frm": b.frm, "to": b.to}

    from .models import AdjustmentRequest
    adj_req = AdjustmentRequest(
        name=req.name, known=list(req.known),
        stations=[p.name for p in req.points],
        observations=obs_models, baselines=bl_models,
        units=req.units, accuracy=req.accuracy,
        confidence=req.confidence, save=False,
    )
    if not obs_models and not bl_models:
        raise AdjustmentError(
            "请求未包含任何候选观测或候选 GNSS 基线", "empty_observation")

    obs_recs, _ = _normalized_observations(adj_req)
    bl_recs, _ = _normalized_baselines(adj_req)
    # 整网预检：基准、连通性、基线已知端点高程等（近似坐标阶段即暴露问题）
    info = _validate_network(adj_req, obs_recs, bl_recs)

    # 锁标记（请求字段 + 锁列表，excluded 优先于 required）
    locked_in, locked_out = set(req.lock_included), set(req.lock_excluded)
    for cid, m in meta.items():
        if m["required"]:
            locked_in.add(cid)
        if m["excluded"]:
            locked_out.add(cid)
    overlap = sorted(locked_in & locked_out)
    if overlap:
        raise AdjustmentError(
            "候选同时被锁定入选与锁定剔除: " + ", ".join(overlap),
            "conflicting_lock", {"ids": overlap})

    return {
        "req": adj_req, "design_req": req, "obs": obs_recs, "bls": bl_recs,
        "info": info, "meta": meta,
        "locked_in": locked_in, "locked_out": locked_out,
    }


# ---------------------------------------------------------------------------
# 2. 参数表与设计矩阵
# ---------------------------------------------------------------------------

def _param_table(ctx):
    req = ctx["req"]
    info = ctx["info"]
    pidx: dict[str, dict[str, int]] = {}
    names: list[tuple[str, str]] = []
    h_stations = set(info["height_stations"])
    for st in req.stations:
        pidx[st] = {}
        if info["horiz_active"]:
            pidx[st]["x"] = len(names); names.append((st, "x"))
            pidx[st]["y"] = len(names); names.append((st, "y"))
        if st in h_stations:
            pidx[st]["h"] = len(names); names.append((st, "h"))

    coords = {k: [p.x, p.y] for k, p in info["known"].items()}
    for p in ctx["design_req"].points:
        coords[p.name] = [p.x, p.y]
    heights = {p.name: p.h for p in req.known if p.h is not None}
    for p in ctx["design_req"].points:
        if p.h is not None and p.name in h_stations:
            heights[p.name] = p.h
    return pidx, names, coords, heights, h_stations


def _subset_records(ctx, selected: set[str]):
    obs = [r for r in ctx["obs"] if r["id"] in selected]
    bls = [b for b in ctx["bls"] if b["id"] in selected]
    return obs, bls


def _subset_feasible(ctx, selected, params):
    """基准 + 连通性预检（子集上复用平差模块规则），返回 (ok, rank, N)。"""
    pidx, names, coords, heights = params[:4]
    obs, bls = _subset_records(ctx, selected)
    ns = SimpleNamespace(known=list(ctx["req"].known), stations=list(ctx["req"].stations))
    try:
        _validate_network(ns, obs, bls)
    except AdjustmentError:
        return False, 0, None
    A, W, _ = _build_matrix(obs, bls, pidx, names, coords, heights)
    if A.shape[0] == 0:
        return False, 0, None
    n = len(names)
    N = A.T @ (W[:, None] * A)
    svals = np.linalg.svd(N, compute_uv=False)
    smax = float(svals[0]) if len(svals) else 0.0
    tol = max(A.shape[0], n) * np.finfo(float).eps * smax if smax > 0 else 0.0
    rank = int(np.sum(svals > tol))
    return rank == n, rank, N


def _build_matrix(obs, bls, pidx, names, coords, heights):
    """在近似坐标处线性化，返回白化设计矩阵 A 与对角权 W（弧度/米）。"""
    n = len(names)
    scalar_rows = []
    for r in obs:
        if r["az"] is not None:
            scalar_rows.append((r["id"], "az", r["std_az"], r["weight"]))
        if r["dist"] is not None:
            scalar_rows.append((r["id"], "dist", r["std_dist"], r["weight"]))
        if r["dh"] is not None:
            scalar_rows.append((r["id"], "dh", r["std_dh"], r["weight"]))
    m = len(scalar_rows) + 3 * len(bls)
    A = np.zeros((m, n))
    W = np.zeros(m)
    row_of: dict[str, Any] = {}

    for i, (oid, kind, std, wf) in enumerate(scalar_rows):
        r = next(rr for rr in obs if rr["id"] == oid)
        _, deriv = _model_and_jacobian(
            kind, r["frm"], r["to"], coords, heights, pidx)
        W[i] = wf / (std * std)
        for j, d in deriv.items():
            A[i, j] = d
        row_of.setdefault(oid, []).append({"row": i, "kind": kind, "std": std,
                                           "weight": wf})
    for k, b in enumerate(bls):
        i = len(scalar_rows) + 3 * k
        _, Jd = _baseline_model(b["frm"], b["to"], coords, heights, pidx)
        Jphys = np.zeros((3, n))
        for j, col in Jd.items():
            Jphys[:, j] = col
        scale = math.sqrt(b["weight"])
        A[i:i + 3, :] = b["Linv"] @ Jphys * scale
        W[i:i + 3] = 1.0
        row_of[b["id"]] = {"row": i, "J": Jphys, "C": b["C"],
                           "weight": b["weight"], "frm": b["frm"], "to": b["to"]}
    return A, W, row_of


# ---------------------------------------------------------------------------
# 3. 精度、冗余度、MDB / MDE
# ---------------------------------------------------------------------------

def _lambda0_scalar(alpha: float, power: float) -> float:
    """标量 Baarda 检验的非中心参数 λ0 = z(1−α/2) + z(1−β)。"""
    return float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))


def _lambda0_group(alpha: float, power: float, df: int = 3) -> float:
    """χ²(df) 整体（备择）检验达到功效 1−β 所需的非中心参数（二分反算）。"""
    crit = float(stats.chi2.ppf(1.0 - alpha, df))

    def pwr(lam):
        return float(stats.chi2.sf(crit, df, lam))

    if pwr(0.0) >= power:
        return 0.0
    lo, hi = 0.0, 1.0
    while pwr(hi) < power:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pwr(mid) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _point_block(Ninv, pidx, name, comps=("x", "y")):
    idx = [pidx[name][c] for c in comps if c in pidx.get(name, {})]
    if not idx:
        return None, idx
    return Ninv[np.ix_(idx, idx)], idx


def _ellipse(cxx, cyy, cxy, k_chi):
    tr = cxx + cyy
    root = math.sqrt(max(0.25 * (cxx - cyy) ** 2 + cxy * cxy, 0.0))
    lam1 = max(0.5 * tr + root, 0.0)
    lam2 = max(0.5 * tr - root, 0.0)
    theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy) % math.pi
    return {
        "semi_major_m": math.sqrt(lam1) * math.sqrt(k_chi),
        "semi_minor_m": math.sqrt(lam2) * math.sqrt(k_chi),
        "std_semi_major_m": math.sqrt(lam1),
        "std_semi_minor_m": math.sqrt(lam2),
        "major_azimuth_deg": rad_to_deg(theta),
    }


def _evaluate(selected, ctx, params, lam_s, lam_g, k_chi,
              compute_impacts=True):
    """计算一个入选集合的协方差、逐行冗余度与 MDB（及目标违反度）。"""
    pidx, names, coords, heights, h_stations = params
    obs, bls = _subset_records(ctx, selected)
    A, W, row_of = _build_matrix(obs, bls, pidx, names, coords, heights)
    m, n = A.shape
    N = A.T @ (W[:, None] * A)
    Ninv = np.linalg.inv(N)

    rows = []
    for r in obs:
        for ent in row_of[r["id"]]:
            i, kind, std, wf = ent["row"], ent["kind"], ent["std"], ent["weight"]
            a = A[i, :]
            h_lev = float(W[i] * a @ Ninv @ a)
            red = float(min(max(1.0 - h_lev, 0.0), 1.0))
            r_eff = max(red, _R_FLOOR)
            # 物理单位 MDB：λ0·σ / (√w·√r)（σ 为弧度或米）
            mdb = lam_s * std / (math.sqrt(wf) * math.sqrt(r_eff))
            det = None
            if compute_impacts:
                grad = Ninv @ (W[i] * a)          # ∂x/∂l（物理单位）
                det = _impact_detail(grad * mdb, pidx, names)
            rows.append({
                "id": r["id"], "kind": kind, "row": i,
                "redundancy": red, "mdb": None if red <= 1e-9 else mdb,
                "mdb_effective": mdb, "std": std, "weight": wf,
                "impact": det,
            })

    groups = []
    for b in bls:
        ent = row_of[b["id"]]
        i = ent["row"]
        Bw = A[i:i + 3, :]
        H = Bw @ Ninv @ Bw.T
        Rm = np.eye(3) - H
        Rm = 0.5 * (Rm + Rm.T)
        w_eig, V_eig = np.linalg.eigh(Rm)
        w_eig = np.clip(w_eig, 0.0, 1.0)
        r_min = float(w_eig[0])
        r_avg = float(np.clip(np.trace(Rm) / 3.0, 0.0, 1.0))
        rmax = max(float(w_eig[-1]), _R_FLOOR)
        # 白化空间中沿 R 最大特征方向的 MDB（欧氏模）
        mdb_w = math.sqrt(lam_g / rmax)
        mdb_phys = None
        impact = None
        if compute_impacts:
            e = V_eig[:, -1]
            dl_w = math.sqrt(lam_g / rmax) * e
            scale = math.sqrt(ent["weight"])
            # 物理粗差向量 = L·∇l_w / √w（白化 l_w = √w L⁻¹ l）
            L = np.linalg.cholesky(ent["C"])
            dl_phys = (L @ dl_w) / scale
            mdb_phys = float(np.linalg.norm(dl_phys))
            grad_x = Ninv @ Bw.T @ dl_w     # 坐标影响（米）
            impact = _impact_detail(grad_x, pidx, names)
        groups.append({
            "id": b["id"], "row": i,
            "redundancy_min": r_min, "redundancy": r_avg,
            "redundancy_eigenvalues": [float(x) for x in w_eig],
            "mdb_whitened_m": mdb_w, "mdb_m": mdb_phys, "impact": impact,
        })

    points = _point_metrics(ctx, Ninv, pidx, k_chi, h_stations)
    return {
        "A": A, "W": W, "N": N, "Ninv": Ninv, "m": m, "n": n,
        "dof": m - n, "rows": rows, "groups": groups, "points": points,
    }


def _impact_detail(dx, pidx, names):
    """把参数空间的坐标影响向量按站点汇总为平面/高程分量。"""
    per_point = {}
    for st, comps in pidx.items():
        vx = vy = vh = 0.0
        if "x" in comps:
            vx = float(dx[comps["x"]])
        if "y" in comps:
            vy = float(dx[comps["y"]])
        if "h" in comps:
            vh = float(dx[comps["h"]])
        if not comps:
            continue
        planar = math.hypot(vx, vy) if ("x" in comps) else None
        total = math.sqrt(
            (vx ** 2 + vy ** 2 if "x" in comps else 0.0) + vh ** 2)
        per_point[st] = {"name": st, "dx_m": vx if "x" in comps else None,
                         "dy_m": vy if "y" in comps else None,
                         "dh_m": vh if "h" in comps else None,
                         "planar_m": planar, "total_m": total}
    worst = max(
        (p["total_m"] for p in per_point.values()), default=0.0)
    return {"per_point": list(per_point.values()),
            "max_coordinate_impact_m": worst}


def _point_metrics(ctx, Ninv, pidx, k_chi, h_stations):
    k_chi3 = float(stats.chi2.ppf(ctx["design_req"].confidence, df=3))
    out = {}
    for st in ctx["req"].stations:
        e: dict[str, Any] = {"name": st}
        if "x" in pidx.get(st, {}):
            ix, iy = pidx[st]["x"], pidx[st]["y"]
            D = Ninv[np.ix_([ix, iy], [ix, iy])]
            cxx, cyy, cxy = float(D[0, 0]), float(D[1, 1]), float(D[0, 1])
            e["std_x_m"] = math.sqrt(max(cxx, 0.0))
            e["std_y_m"] = math.sqrt(max(cyy, 0.0))
            e["std_horizontal_m"] = math.sqrt(max(cxx + cyy, 0.0))
            e["ellipse"] = _ellipse(cxx, cyy, cxy, k_chi)
            e["ellipse"]["confidence"] = ctx["design_req"].confidence
            e["covariance_xy_m2"] = [[cxx, cxy], [cxy, cyy]]
        if "h" in pidx.get(st, {}):
            ih = pidx[st]["h"]
            e["std_h_m"] = math.sqrt(max(float(Ninv[ih, ih]), 0.0))
        if "x" in pidx.get(st, {}) and "h" in pidx.get(st, {}):
            idx = [pidx[st]["x"], pidx[st]["y"], pidx[st]["h"]]
            D3 = Ninv[np.ix_(idx, idx)]
            e["covariance_xyz_m2"] = [
                [float(x) for x in row] for row in D3]
            w3, V3 = np.linalg.eigh(0.5 * (D3 + D3.T))
            w3 = np.clip(w3, 0.0, None)
            order = np.argsort(w3)[::-1]
            axes3 = [float(math.sqrt(max(w3[k], 0.0)) * math.sqrt(k_chi3))
                     for k in order]
            major = V3[:, order[0]]
            e["error_ellipsoid"] = {
                "semi_axes_m": axes3,
                "major_axis_azimuth_deg": float(
                    rad_to_deg(math.atan2(major[1], major[0])) % 360.0),
                "major_axis_dip_deg": float(
                    rad_to_deg(math.asin(float(np.clip(major[2], -1.0, 1.0))))),
                "confidence": ctx["design_req"].confidence,
                "chi2_scale_df3": k_chi3,
            }
        out[st] = e
    return out


# ---------------------------------------------------------------------------
# 4. 目标违反度与贪心选择
# ---------------------------------------------------------------------------

def _objective(ev, ctx, goals: AccuracyGoals):
    """无量纲目标违反度（各约束 ratio−1 的正部之和），返回 (值, 明细)。"""
    val = 0.0
    detail = {"points": [], "reliability": [], "mdb": []}
    target_points = set(goals.points) if goals.points else \
        set(ctx["req"].stations)

    for name in sorted(target_points):
        p = ev["points"].get(name)
        if p is None:
            continue
        checks = []
        if goals.max_horizontal_std_m is not None and "std_horizontal_m" in p:
            checks.append(("horizontal_std", p["std_horizontal_m"],
                           goals.max_horizontal_std_m))
        if goals.max_ellipse_semi_major_m is not None and "ellipse" in p:
            checks.append(("ellipse_semi_major",
                           p["ellipse"]["semi_major_m"],
                           goals.max_ellipse_semi_major_m))
        if goals.max_height_std_m is not None and "std_h_m" in p:
            checks.append(("height_std", p["std_h_m"],
                           goals.max_height_std_m))
        for dim, value, limit in checks:
            ratio = value / limit
            if ratio > 1.0:
                val += ratio - 1.0
                detail["points"].append(
                    {"point": name, "dimension": dim,
                     "value_m": value, "limit_m": limit, "ratio": ratio})

    if goals.min_redundancy is not None:
        rmin = goals.min_redundancy
        for row in ev["rows"]:
            if row["redundancy"] < rmin:
                ratio = rmin / max(row["redundancy"], _R_FLOOR)
                val += min(ratio - 1.0, 1e6)
                detail["reliability"].append(
                    {"id": row["id"], "component": _kind_name(row["kind"]),
                     "redundancy": row["redundancy"], "limit": rmin})
        for g in ev["groups"]:
            if g["redundancy_min"] < rmin:
                ratio = rmin / max(g["redundancy_min"], _R_FLOOR)
                val += min(ratio - 1.0, 1e6)
                detail["reliability"].append(
                    {"id": g["id"], "component": "gnss_baseline",
                     "redundancy": g["redundancy_min"], "limit": rmin})

    mdb_limits = [
        ("az", goals.max_mdb_azimuth_sec, True),
        ("dist", goals.max_mdb_distance_m, False),
        ("dh", goals.max_mdb_height_m, False),
    ]
    for row in ev["rows"]:
        for kind, limit, arcsec in mdb_limits:
            if limit is None or row["kind"] != kind:
                continue
            mdb = row["mdb_effective"]
            value = rad_to_deg(mdb) * 3600.0 if arcsec else mdb
            ratio = value / limit
            if ratio > 1.0:
                val += min(ratio - 1.0, 1e6)
                detail["mdb"].append(
                    {"id": row["id"], "component": _kind_name(kind),
                     "mdb": value, "limit": limit, "ratio": min(ratio, 1e7)})
    if goals.max_mdb_baseline_m is not None:
        for g in ev["groups"]:
            ratio = g["mdb_whitened_m"] / goals.max_mdb_baseline_m
            if ratio > 1.0:
                val += min(ratio - 1.0, 1e6)
                detail["mdb"].append(
                    {"id": g["id"], "component": "gnss_baseline",
                     "mdb": g["mdb_whitened_m"],
                     "limit": goals.max_mdb_baseline_m,
                     "ratio": min(ratio, 1e7)})
    return val, detail


def _kind_name(kind):
    return {"az": "azimuth", "dist": "distance", "dh": "height_difference"}[kind]


def _cost(ctx, selected):
    return sum(ctx["meta"][cid]["cost"] for cid in selected)


def run_design(req: NetworkDesignRequest) -> dict[str, Any]:
    """执行测前网形设计主流程。"""
    ctx = _build_context(req)
    params = _param_table(ctx)
    pidx, names = params[0], params[1]
    n = len(names)
    goals = req.goals

    lam_s = _lambda0_scalar(req.significance_alpha, req.power)
    lam_g = _lambda0_group(req.significance_alpha, req.power, df=3)
    k_chi = float(stats.chi2.ppf(req.confidence, df=2))

    available = set(ctx["meta"]) - ctx["locked_out"]
    selected = set(ctx["locked_in"] & available)
    required_cost = _cost(ctx, selected)
    if req.budget is not None and required_cost > req.budget + 1e-9:
        raise AdjustmentError(
            f"必选观测成本 {required_cost:g} 已超过预算 {req.budget:g}",
            "budget_exceeded",
            {"required_cost": required_cost, "budget": req.budget,
             "required_ids": sorted(selected)},
        )

    selection_log: list[dict[str, Any]] = []
    warnings: list[str] = []

    # ---- 阶段 1：结构（基准/连通/秩）----
    def rank_of(s):
        ok, rank, _ = _subset_feasible(ctx, s, *params)
        return rank if ok else -1

    rank = rank_of(selected) if selected else 0
    if rank < 0:
        rank = 0
    guard = 0
    while rank < n:
        guard += 1
        if guard > n + len(available) + 2:
            raise AdjustmentError("结构选择异常终止", "design_infeasible")
        trials = []
        for cid in sorted(available - selected):
            cost = _cost(ctx, selected | {cid})
            if req.budget is not None and cost > req.budget + 1e-9:
                continue
            r = rank_of(selected | {cid})
            if r > rank:
                trials.append((r, ctx["meta"][cid]["cost"], cid))
        if not trials:
            # 即便超预算也要给出“理论上能补秩”的候选，便于错误信息定位
            all_trials = []
            for cid in sorted(available - selected):
                r = rank_of(selected | {cid})
                if r > rank:
                    all_trials.append((r, ctx["meta"][cid]["cost"], cid))
            if not all_trials:
                _raise_infeasible(ctx, selected, params, goals)
            cheapest = _cost(ctx, selected) + min(
                ctx["meta"][c]["cost"] for _, _, c in all_trials)
            raise AdjustmentError(
                f"预算 {req.budget:g} 不足以使网形达到满秩（至少还需成本 "
                f"{cheapest - _cost(ctx, selected):g}）",
                "budget_exceeded",
                {"budget": req.budget, "spent": _cost(ctx, selected),
                 "minimum_extra_cost": cheapest - _cost(ctx, selected),
                 "rank": rank, "required_rank": n})
        trials.sort(key=lambda t: (-t[0], t[1], t[2]))
        new_rank, _, cid = trials[0]
        selected.add(cid)
        selection_log.append({
            "order": len(selection_log) + 1, "id": cid,
            "type": ctx["meta"][cid]["kind"], "stage": "structure",
            "from": ctx["meta"][cid]["frm"], "to": ctx["meta"][cid]["to"],
            "cost": ctx["meta"][cid]["cost"],
            "cumulative_cost": _cost(ctx, selected),
            "rank_delta": new_rank - rank,
            "marginal_gain": None,
        })
        rank = new_rank

    # ---- 阶段 2：按边际精度收益贪心 ----
    ev = _evaluate(selected, ctx, params, lam_s, lam_g, k_chi,
                   compute_impacts=False)
    obj, _ = _objective(ev, ctx, goals)
    stop_reason = "targets_met" if obj <= _GAIN_EPS else None
    while obj > _GAIN_EPS:
        candidates = []
        for cid in sorted(available - selected):
            extra = ctx["meta"][cid]["cost"]
            if req.budget is not None and \
                    _cost(ctx, selected) + extra > req.budget + 1e-9:
                continue
            trial = _evaluate(selected | {cid}, ctx, params, lam_s, lam_g,
                              k_chi, compute_impacts=False)
            tobj, _ = _objective(trial, ctx, goals)
            candidates.append((obj - tobj, extra, cid))
        if not candidates:
            stop_reason = "budget_exhausted"
            break
        best_gain = max(c[0] for c in candidates)
        if best_gain <= _GAIN_EPS:
            stop_reason = "no_marginal_gain"
            break
        # 收益最大；并列时成本低、id 小者优先
        gain, extra, cid = sorted(
            [c for c in candidates if c[0] >= best_gain - 1e-12],
            key=lambda t: (t[1], t[2]))[0]
        selected.add(cid)
        selection_log.append({
            "order": len(selection_log) + 1, "id": cid,
            "type": ctx["meta"][cid]["kind"], "stage": "precision",
            "from": ctx["meta"][cid]["frm"], "to": ctx["meta"][cid]["to"],
            "cost": extra, "cumulative_cost": _cost(ctx, selected),
            "rank_delta": 0, "marginal_gain": gain,
        })
        ev = _evaluate(selected, ctx, params, lam_s, lam_g, k_chi,
                       compute_impacts=False)
        obj, _ = _objective(ev, ctx, goals)
    if stop_reason is None:
        stop_reason = "targets_met"

    # ---- 最终完整评定（含 MDE/坐标影响）与未选原因 ----
    final = _evaluate(selected, ctx, params, lam_s, lam_g, k_chi,
                      compute_impacts=True)
    fobj, violations = _objective(final, ctx, goals)
    targets_met = fobj <= _GAIN_EPS

    unselected = []
    for cid, m in sorted(ctx["meta"].items()):
        if cid in selected:
            continue
        affordable = req.budget is None or \
            _cost(ctx, selected) + m["cost"] <= req.budget + 1e-9
        if cid in ctx["locked_out"]:
            reason, gain = "locked_excluded", None
        elif not affordable:
            reason, gain = "budget", None
        else:
            trial = _evaluate(selected | {cid}, ctx, params, lam_s,
                              lam_g, k_chi, compute_impacts=False)
            tobj, _ = _objective(trial, ctx, goals)
            gain = fobj - tobj
            if targets_met and stop_reason == "targets_met":
                reason = "targets_met"
            elif gain <= _GAIN_EPS:
                reason = "no_marginal_gain"
            else:
                # 理论上有收益却未选（预算耗尽或循环提前终止）
                reason = "not_selected"
        unselected.append({
            "id": cid, "type": m["kind"], "from": m["frm"], "to": m["to"],
            "cost": m["cost"], "reason": reason,
            "marginal_gain": gain,
        })

    if not targets_met:
        warnings.append(
            "精度/可靠性目标未能全部满足（停止原因："
            + {"budget_exhausted": "预算不足",
               "no_marginal_gain": "剩余候选无边际精度收益"}.get(
                stop_reason, stop_reason) + "），见 unmet_targets")
    if final["dof"] == 0:
        warnings.append("入选网形多余观测数为 0：无校核条件，冗余度与 MDB 仅作极限提示")

    result = _assemble_result(
        req, ctx, final, selected, selection_log, unselected,
        violations, targets_met, stop_reason, warnings,
        lam_s, lam_g, params)
    return result


def _raise_infeasible(ctx, selected, params, goals):
    """满秩不可达：给出基准/连通/秩亏诊断。"""
    obs, bls = _subset_records(ctx, set(ctx["meta"]) - ctx["locked_out"])
    ns = SimpleNamespace(known=list(ctx["req"].known),
                         stations=list(ctx["req"].stations))
    try:
        _validate_network(ns, obs, bls)
    except AdjustmentError as e:
        raise AdjustmentError(
            "全部候选观测仍无法构成可行网形：" + str(e), e.code, e.details)
    A, W, _ = _build_matrix(obs, bls, *params[:4])
    n = A.shape[1]
    N = A.T @ (W[:, None] * A)
    svals = np.linalg.svd(N, compute_uv=False)
    smax = float(svals[0]) if len(svals) else 0.0
    tol = max(A.shape[0], n) * np.finfo(float).eps * smax if smax > 0 else 0.0
    rank = int(np.sum(svals > tol))
    d = _rank_deficiency_details(N, params[1], rank)
    raise AdjustmentError(
        f"全部候选观测仍无法确定全部参数（缺秩 {n - rank} 维）："
        f"{d['description']}；请补充候选观测或增加已知点",
        "design_infeasible", d)


# ---------------------------------------------------------------------------
# 5. 结果组装与方案对比
# ---------------------------------------------------------------------------

def _assemble_result(req, ctx, ev, selected, selection_log, unselected,
                     violations, targets_met, stop_reason, warnings,
                     lam_s, lam_g, params):
    pidx = params[0]
    total_cost = _cost(ctx, selected)

    # 入选观测的逐分量可靠性 + MDB/MDE
    obs_out, bl_out = [], []
    obs_by_id = {r["id"]: r for r in ctx["obs"]}
    rows_by_id: dict[str, list] = {}
    for row in ev["rows"]:
        rows_by_id.setdefault(row["id"], []).append(row)
    for cid in sorted(selected):
        m = ctx["meta"][cid]
        if m["kind"] != "observation":
            continue
        comps = []
        for row in rows_by_id.get(cid, []):
            item = {
                "component": _kind_name(row["kind"]),
                "redundancy": row["redundancy"],
                "leverage": 1.0 - row["redundancy"],
                "mdb": _mdb_with_unit(row),
                "coordinate_impact_m": row["impact"],
            }
            comps.append(item)
        reds = [c["redundancy"] for c in comps]
        obs_out.append({
            "id": cid, "from": m["frm"], "to": m["to"],
            "required": cid in ctx["locked_in"],
            "cost": m["cost"], "redundancy": min(reds),
            "components": comps,
        })
    group_by_id = {g["id"]: g for g in ev["groups"]}
    for cid in sorted(selected):
        m = ctx["meta"][cid]
        if m["kind"] != "baseline":
            continue
        g = group_by_id[cid]
        bl_out.append({
            "id": cid, "from": m["frm"], "to": m["to"],
            "required": cid in ctx["locked_in"],
            "cost": m["cost"],
            "redundancy": g["redundancy"],
            "min_redundancy_eigenvalue": g["redundancy_min"],
            "redundancy_eigenvalues": g["redundancy_eigenvalues"],
            "mdb_whitened_m": g["mdb_whitened_m"],
            "mdb_m": g["mdb_m"],
            "coordinate_impact_m": g["impact"],
        })

    points = []
    for st in ctx["req"].stations:
        e = ev["points"][st]
        ap = ctx["design_req"].points
        approx = next(p for p in ap if p.name == st)
        e = dict(e)
        e["x"] = approx.x
        e["y"] = approx.y
        e["h"] = approx.h
        points.append(e)

    weakest = _weakest(ev, ctx)
    unmet_points = sorted({v["point"] for v in violations["points"]})

    result = {
        "status": "ok",
        "name": req.name,
        "algorithm_version": DESIGN_ALGORITHM_VERSION,
        "software_version": __version__,
        "units": req.units.model_dump(),
        "parameters": {
            "significance_alpha": req.significance_alpha,
            "power": req.power,
            "confidence": req.confidence,
            "budget": req.budget,
            "lambda0_scalar": lam_s,
            "lambda0_baseline_group": lam_g,
            "z_alpha_two_sided": float(
                stats.norm.ppf(1.0 - req.significance_alpha / 2.0)),
            "z_beta": float(stats.norm.ppf(req.power)),
        },
        "datum": {
            "known_points": [p.model_dump() for p in req.known],
            "horizontal_active": ctx["info"]["horiz_active"],
            "height_active": ctx["info"]["height_active"],
        },
        "selected_order": selection_log,
        "selected_observations": obs_out,
        "selected_baselines": bl_out,
        "unselected": unselected,
        "points": points,
        "statistics": {
            "parameters_n": ev["n"],
            "rank": ev["n"],
            "rank_deficiency": 0,
            "observation_rows": ev["m"],
            "scalar_rows": len(ev["rows"]),
            "baseline_rows": 3 * len(ev["groups"]),
            "degrees_of_freedom": ev["dof"],
            "selected_observation_count": len(obs_out),
            "selected_baseline_count": len(bl_out),
            "total_cost": total_cost,
            "budget": req.budget,
            "budget_remaining": (req.budget - total_cost
                                 if req.budget is not None else None),
            "stop_reason": stop_reason,
            "targets_met": targets_met,
            "min_redundancy": weakest["min_redundancy"],
            "weakest_point": weakest["point"],
            "worst_mdb": weakest["mdb"],
        },
        "goals": req.goals.model_dump(),
        "unmet_targets": violations,
        "unmet_points": unmet_points,
        "warnings": warnings,
    }
    return result


def _mdb_with_unit(row):
    kind, mdb = row["kind"], row["mdb"]
    if mdb is None:
        return {"value": None, "unit": None, "detectable": False}
    if kind == "az":
        return {"value": rad_to_deg(mdb) * 3600.0, "unit": "arcsec",
                "detectable": True}
    return {"value": mdb, "unit": "m", "detectable": True}


def _weakest(ev, ctx):
    weak_pt = None
    for name, p in ev["points"].items():
        cand = (p.get("std_horizontal_m"), name, "horizontal")
        if cand[0] is not None and (weak_pt is None or cand[0] > weak_pt[0]):
            weak_pt = cand
        if p.get("std_h_m") is not None and \
                (weak_pt is None or p["std_h_m"] > weak_pt[0]):
            weak_pt = (p["std_h_m"], name, "height")
    point_info = None
    if weak_pt is not None:
        point_info = {"point": weak_pt[1], "dimension": weak_pt[2],
                      "std_m": weak_pt[0]}

    rmin, rid, rcomp = 1.0, None, None
    for row in ev["rows"]:
        if row["redundancy"] < rmin:
            rmin, rid, rcomp = row["redundancy"], row["id"], _kind_name(row["kind"])
    for g in ev["groups"]:
        if g["redundancy_min"] < rmin:
            rmin, rid, rcomp = g["redundancy_min"], g["id"], "gnss_baseline"

    worst = {}
    for kind in ("az", "dist", "dh"):
        vals = [r for r in ev["rows"] if r["kind"] == kind and r["mdb"] is not None]
        if vals:
            r = max(vals, key=lambda x: x["mdb"])
            worst[_kind_name(kind)] = {
                "id": r["id"],
                "mdb": (rad_to_deg(r["mdb"]) * 3600.0 if kind == "az"
                        else r["mdb"]),
                "unit": "arcsec" if kind == "az" else "m",
            }
    if ev["groups"]:
        g = max(ev["groups"], key=lambda x: x["mdb_whitened_m"])
        worst["gnss_baseline"] = {"id": g["id"],
                                  "mdb": g["mdb_whitened_m"], "unit": "m"}
    return {"point": point_info,
            "min_redundancy": {"value": rmin if (ev["rows"] or ev["groups"])
                               else None, "id": rid, "component": rcomp},
            "mdb": worst}


def design_summary(result: dict[str, Any]) -> dict[str, Any]:
    """供存储摘要与方案对比使用的精简指标。"""
    s = result["statistics"]
    return {
        "name": result.get("name"),
        "algorithm_version": result.get("algorithm_version"),
        "total_cost": s["total_cost"],
        "rank": s["rank"],
        "rank_deficiency": s["rank_deficiency"],
        "parameters_n": s["parameters_n"],
        "degrees_of_freedom": s["degrees_of_freedom"],
        "selected_observation_count": s["selected_observation_count"],
        "selected_baseline_count": s["selected_baseline_count"],
        "targets_met": s["targets_met"],
        "stop_reason": s["stop_reason"],
        "unmet_points": result.get("unmet_points"),
        "weakest_point": s["weakest_point"],
        "min_redundancy": s["min_redundancy"],
        "worst_mdb": s["worst_mdb"],
        "selected_ids": [e["id"] for e in result["selected_order"]],
    }


def compare_designs(result_a: dict[str, Any],
                    result_b: dict[str, Any]) -> dict[str, Any]:
    """对比两套网形：成本、最弱点精度、可靠性、秩及入选集合差异。"""
    sa, sb = design_summary(result_a), design_summary(result_b)

    def weak_pt(s):
        wp = s["weakest_point"]
        return None if wp is None else wp["std_m"]

    ids_a = set(sa["selected_ids"])
    ids_b = set(sb["selected_ids"])
    scalar_keys = ("total_cost", "rank", "rank_deficiency", "parameters_n",
                   "degrees_of_freedom", "selected_observation_count",
                   "selected_baseline_count")
    differences = {
        k: {"a": sa[k], "b": sb[k], "delta_b_minus_a":
            (sb[k] - sa[k]) if isinstance(sa[k], (int, float)) else None}
        for k in scalar_keys
    }
    wa, wb = weak_pt(sa), weak_pt(sb)
    differences["weakest_point_std_m"] = {
        "a": wa, "b": wb,
        "delta_b_minus_a": (wb - wa) if wa is not None and wb is not None else None,
    }
    ra = sa["min_redundancy"]["value"] if sa["min_redundancy"] else None
    rb = sb["min_redundancy"]["value"] if sb["min_redundancy"] else None
    differences["min_redundancy"] = {
        "a": ra, "b": rb,
        "delta_b_minus_a": (rb - ra) if ra is not None and rb is not None else None,
    }
    return {
        "status": "ok",
        "design_a": sa,
        "design_b": sb,
        "differences": differences,
        "selection_diff": {
            "only_in_a": sorted(ids_a - ids_b),
            "only_in_b": sorted(ids_b - ids_a),
            "common": sorted(ids_a & ids_b),
        },
        "targets": {
            "a_met": sa["targets_met"], "b_met": sb["targets_met"],
            "a_unmet_points": sa["unmet_points"],
            "b_unmet_points": sb["unmet_points"],
        },
    }


# ---------------------------------------------------------------------------
# 6. 历史版本重演（加锁 / 解锁 / 调预算）
# ---------------------------------------------------------------------------

def apply_replay_options(req: NetworkDesignRequest,
                         options: Optional[DesignReplayOptions]) -> NetworkDesignRequest:
    req = req.model_copy(deep=True)
    if options is None:
        return req
    if options.budget is not None:
        req.budget = options.budget
    req.lock_included = list(req.lock_included) + list(options.lock_included)
    req.lock_excluded = list(req.lock_excluded) + list(options.lock_excluded)
    unlock = set(options.unlock_required)
    for o in req.observations:
        if o.id in unlock:
            o.required = False
    for b in req.baselines:
        if b.id in unlock:
            b.required = False
    return req
