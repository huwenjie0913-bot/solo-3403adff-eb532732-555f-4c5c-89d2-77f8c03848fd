"""测量网（平面导线 + 高程 + GNSS 三维基线）加权最小二乘平差核心。

函数模型（每条有观测的边产生若干观测方程行）：

* 方位角  a = atan2(Δy, Δx)
* 平距    s = sqrt(Δx² + Δy²)
* 高差    dh = h(to) - h(from)
* GNSS 基线 [dx, dy, dh]ᵀ = X(to) - X(from)，三分量相关，
  协方差阵为完整 3×3 矩阵；平差前用 Cholesky 分解同时白化
  残差与雅可比（b = L⁻¹·l，B = L⁻¹·J），白化后按单位权参与法方程

内部统一 SI 单位（弧度、米）。参数向量由各待求站点的 x/y/h 组成，
Gauss-Newton 迭代求解  N·dx = AᵀWl。
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

import numpy as np
from scipy import stats

from .models import AdjustmentRequest
from .units import (
    angle_diff,
    angle_to_rad,
    distance_to_m,
    m_to_distance,
    normalize_angle,
    rad_to_deg,
    rad_to_dms,
    rad_to_gon,
)

TWO_PI = 2.0 * math.pi
ARCSEC = math.pi / (180.0 * 3600.0)


class AdjustmentError(Exception):
    """预检失败（400 类错误）。"""

    def __init__(self, message: str, code: str = "adjustment_error", details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


# ---------------------------------------------------------------------------
# 1. 观测归一化与预检
# ---------------------------------------------------------------------------

def _normalized_observations(req: AdjustmentRequest):
    obs: list[dict[str, Any]] = []
    seen_keys: dict[tuple, list[str]] = defaultdict(list)
    for i, o in enumerate(req.observations, start=1):
        oid = o.id or f"o{i}"
        if o.frm == o.to:
            raise AdjustmentError(
                f"观测 {oid} 的测站与照准点相同（{o.frm}），不允许自环边",
                "self_loop", {"id": oid},
            )
        rec: dict[str, Any] = {
            "id": oid, "frm": o.frm, "to": o.to,
            "weight": float(o.weight), "index": i - 1,
        }
        # 方位角
        if o.azimuth is not None:
            rec["az"] = normalize_angle(angle_to_rad(o.azimuth, req.units.angle))
            if o.std_azimuth is not None:
                if req.units.angle == "dms":
                    rec["std_az"] = abs(float(o.std_azimuth)) * ARCSEC
                else:
                    rec["std_az"] = abs(angle_to_rad(o.std_azimuth, req.units.angle))
            else:
                rec["std_az"] = req.accuracy.std_azimuth_sec * ARCSEC
            seen_keys[(o.frm, o.to, "az")].append(oid)
        else:
            rec["az"] = rec["std_az"] = None
        # 距离
        if o.distance is not None:
            s = distance_to_m(o.distance, req.units.distance)
            rec["dist"] = s
            if o.std_distance is not None:
                rec["std_dist"] = distance_to_m(o.std_distance, req.units.distance)
            else:
                c = req.accuracy.std_dist_const
                ppm = req.accuracy.std_dist_ppm * 1e-6 * s
                rec["std_dist"] = math.hypot(c, ppm)
            seen_keys[(o.frm, o.to, "dist")].append(oid)
        else:
            rec["dist"] = rec["std_dist"] = None
        # 高差
        if o.dh is not None:
            rec["dh"] = distance_to_m(float(o.dh), req.units.height)
            rec["std_dh"] = (
                distance_to_m(o.std_dh, req.units.height)
                if o.std_dh is not None else float(req.accuracy.std_dh)
            )
            seen_keys[(o.frm, o.to, "dh")].append(oid)
        else:
            rec["dh"] = rec["std_dh"] = None

        if rec["az"] is None and rec["dist"] is None and rec["dh"] is None:
            raise AdjustmentError(
                f"观测 {oid} 没有任何有效分量（azimuth/distance/dh 至少给一项）",
                "empty_observation", {"id": oid},
            )
        for k_std in ("std_az", "std_dist", "std_dh"):
            v = rec[k_std]
            if v is not None and not v > 0.0:
                raise AdjustmentError(
                    f"观测 {oid} 的标准差必须为正数", "bad_std", {"id": oid}
                )
        obs.append(rec)

    duplicates = [
        {"key": f"{k[0]}->{k[1]} {k[2]}", "ids": ids}
        for k, ids in seen_keys.items() if len(ids) > 1
    ]
    return obs, duplicates


def _normalized_baselines(req: AdjustmentRequest):
    """GNSS 基线归一化：单位换算 + 协方差对称/正定检查 + Cholesky 因子。"""
    bls: list[dict[str, Any]] = []
    seen_keys: dict[tuple, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    for i, b in enumerate(req.baselines, start=1):
        bid = b.id or f"b{i}"
        if bid in seen_ids:
            raise AdjustmentError(
                f"基线编号重复: {bid}", "duplicate_point", {"id": bid})
        seen_ids.add(bid)
        if b.frm == b.to:
            raise AdjustmentError(
                f"基线 {bid} 的起终点相同（{b.frm}），不允许自环边",
                "self_loop", {"id": bid},
            )
        if not (b.weight > 0.0):
            raise AdjustmentError(
                f"基线 {bid} 的权重因子必须为正数（相关观测不允许零权）",
                "bad_std", {"id": bid},
            )
        f_unit = distance_to_m(1.0, b.unit or req.units.distance)
        try:
            C = np.asarray(b.covariance, dtype=float)
        except (TypeError, ValueError):
            raise AdjustmentError(
                f"基线 {bid} 的协方差必须是 3×3 数值矩阵",
                "bad_covariance_shape", {"id": bid, "shape": list(np.shape(b.covariance))},
            )
        if C.shape != (3, 3):
            raise AdjustmentError(
                f"基线 {bid} 的协方差必须是完整 3×3 矩阵（当前形状 {C.shape}）",
                "bad_covariance_shape",
                {"id": bid, "shape": list(C.shape)},
            )
        if not np.all(np.isfinite(C)):
            raise AdjustmentError(
                f"基线 {bid} 的协方差含有非数值（NaN/Inf）",
                "bad_covariance_value", {"id": bid},
            )
        asym = float(np.max(np.abs(C - C.T)))
        if asym > 1e-9 * max(1.0, float(np.max(np.abs(C)))):
            raise AdjustmentError(
                f"基线 {bid} 的协方差阵不对称（最大不对称量 {asym:.3e} m²）；"
                "接收机协方差必须为对称阵，请核对上下三角",
                "covariance_not_symmetric",
                {"id": bid, "max_asymmetry_m2": asym},
            )
        C = 0.5 * (C + C.T) * f_unit**2
        if np.any(np.diag(C) <= 0.0):
            raise AdjustmentError(
                f"基线 {bid} 的协方差对角元素必须严格为正",
                "covariance_not_positive_definite",
                {"id": bid, "diagonal_m2": [float(x) for x in np.diag(C)]},
            )
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            eig = np.linalg.eigvalsh(C)
            raise AdjustmentError(
                f"基线 {bid} 的协方差阵非正定（最小特征值 "
                f"{float(eig.min()):.3e} m²），无法进行 Cholesky 白化",
                "covariance_not_positive_definite",
                {"id": bid,
                 "min_eigenvalue_m2": float(eig.min()),
                 "eigenvalues_m2": [float(x) for x in eig]},
            )
        obs_vec = f_unit * np.array([b.dx, b.dy, b.dh], dtype=float)
        if not np.all(np.isfinite(obs_vec)):
            raise AdjustmentError(
                f"基线 {bid} 的 dx/dy/dh 含有非数值", "bad_std", {"id": bid},
            )
        bls.append({
            "id": bid, "frm": b.frm, "to": b.to,
            "vec": obs_vec, "C": C, "L": L, "Linv": np.linalg.inv(L),
            "weight": float(b.weight), "unit": b.unit or req.units.distance,
            "unit_factor": f_unit, "index": i - 1,
        })
        seen_keys[(b.frm, b.to)].append(bid)

    duplicates = [
        {"key": f"{k[0]}->{k[1]} gnss_baseline", "ids": ids}
        for k, ids in seen_keys.items() if len(ids) > 1
    ]
    return bls, duplicates


def _validate_network(req, obs, bls):
    warnings: list[str] = []

    known = {}
    for p in req.known:
        if p.name in known:
            raise AdjustmentError(f"已知点重名: {p.name}", "duplicate_point")
        known[p.name] = p
    station_set = set()
    for s in req.stations:
        if s in known:
            raise AdjustmentError(f"站点 {s} 同时出现在已知点中", "duplicate_point")
        if s in station_set:
            raise AdjustmentError(f"待求站点重名: {s}", "duplicate_point")
        station_set.add(s)

    # 观测 id / 基线 id 不允许互相冲突
    obs_ids = {r["id"] for r in obs}
    bl_ids = {b["id"] for b in bls}
    clash = sorted(obs_ids & bl_ids)
    if clash:
        raise AdjustmentError(
            "观测与基线编号冲突: " + ", ".join(clash),
            "duplicate_point", {"ids": clash},
        )

    all_names = set(known) | station_set
    for r in obs:
        for end in (r["frm"], r["to"]):
            if end not in all_names:
                raise AdjustmentError(
                    f"观测 {r['id']} 的端点 {end} 既不是已知点也不是待求站点",
                    "unknown_endpoint", {"id": r["id"], "endpoint": end},
                )
    for b in bls:
        for end in (b["frm"], b["to"]):
            if end not in all_names:
                raise AdjustmentError(
                    f"基线 {b['id']} 的端点 {end} 既不是已知点也不是待求站点",
                    "unknown_endpoint", {"id": b["id"], "endpoint": end},
                )

    # 基线提供三维信息，同时加入平面图、高程图与基线图
    baseline_edges = {(b["frm"], b["to"]) for b in bls}
    horiz_active = any(r["az"] is not None or r["dist"] is not None for r in obs) \
        or bool(bls)
    height_active = any(r["dh"] is not None for r in obs) or bool(bls)

    h_adj, v_adj, g_adj = (
        defaultdict(set), defaultdict(set), defaultdict(set))
    for r in obs:
        if r["az"] is not None or r["dist"] is not None:
            h_adj[r["frm"]].add(r["to"])
            h_adj[r["to"]].add(r["frm"])
        if r["dh"] is not None:
            v_adj[r["frm"]].add(r["to"])
            v_adj[r["to"]].add(r["frm"])
    for b in bls:
        h_adj[b["frm"]].add(b["to"]); h_adj[b["to"]].add(b["frm"])
        v_adj[b["frm"]].add(b["to"]); v_adj[b["to"]].add(b["frm"])
        g_adj[b["frm"]].add(b["to"]); g_adj[b["to"]].add(b["frm"])

    def components(adj, nodes):
        seen, comps = set(), []
        for root in sorted(nodes):
            if root in seen:
                continue
            q, comp = deque([root]), []
            seen.add(root)
            while q:
                u = q.popleft()
                comp.append(u)
                for w in sorted(adj[u]):
                    if w not in seen:
                        seen.add(w)
                        q.append(w)
            comps.append(comp)
        return comps

    def touches_baseline(comp):
        nodes = set(comp)
        for u, vs in g_adj.items():
            if u in nodes and vs:
                return True
        return False

    if horiz_active:
        if len(known) < 1:
            raise AdjustmentError(
                "平面网基准不足：至少需要 1 个已知坐标点"
                "（纯全站仪网需要 2 个以固定旋转）",
                "insufficient_datum", {"known_count": len(known)},
            )
        comps = components(h_adj, set(h_adj.keys()))
        unreachable = []
        for comp in comps:
            cknown = [n for n in comp if n in known]
            if not cknown:
                unreachable.extend(n for n in comp if n not in known)
        if unreachable:
            raise AdjustmentError(
                "平面网存在不与任何已知点连通的部分: " + ", ".join(sorted(unreachable)),
                "disconnected", {"stations": sorted(unreachable)},
            )
        if not bls:
            # 纯全站仪网（方位角+平距）：全局至少 2 个已知点固定平移与旋转
            if len(known) < 2:
                raise AdjustmentError(
                    "平面网基准不足：全站仪网（方位角+平距）至少需要 2 个已知坐标点"
                    "以固定平移与旋转；含 GNSS 基线时 1 个已知点即可（基线自带方向）",
                    "insufficient_datum", {"known_count": len(known)},
                )
        else:
            # 含 GNSS 基线：基线自带尺度与方向，分量含基线时 1 个已知点即可
            # 固定平移；不含基线的全站仪分量仍需 2 个已知点固定旋转
            for comp in comps:
                cknown = [n for n in comp if n in known]
                free = sorted(n for n in comp if n not in known)
                if len(cknown) >= 2 or not free:
                    continue
                if touches_baseline(comp):
                    continue
                raise AdjustmentError(
                    "平面分量 " + ", ".join(free) + " 不含 GNSS 基线且只与 1 个"
                    "已知点相连，网的旋转无法固定（该分量需 2 个已知点，"
                    "或加入 GNSS 基线）",
                    "insufficient_datum", {"stations": free},
                )

    h_known = [p.name for p in req.known if p.h is not None]
    if height_active:
        if not h_known:
            raise AdjustmentError(
                "高程网基准不足：至少需要 1 个已知高程点",
                "insufficient_height_datum",
            )
        unreachable = []
        for comp in components(v_adj, set(v_adj.keys())):
            if not any(n in h_known for n in comp):
                unreachable.extend(n for n in comp if n not in h_known)
        if unreachable:
            raise AdjustmentError(
                "高程网存在不与已知高程点连通的部分: " + ", ".join(sorted(unreachable)),
                "height_disconnected", {"stations": sorted(unreachable)},
            )
        # 基线上的端点必须有高程：已知点必须给 h，待求站点自动增列 h 参数
        for b in bls:
            for end in (b["frm"], b["to"]):
                if end in known and known[end].h is None:
                    raise AdjustmentError(
                        f"基线 {b['id']} 的端点 {end} 是无已知高程的已知点："
                        "三维基线要求其已知端点给出 h（或把该点列为待求站点）",
                        "baseline_endpoint_without_height",
                        {"id": b["id"], "endpoint": end},
                    )

    used = set()
    for r in obs:
        used.add(r["frm"])
        used.add(r["to"])
    for b in bls:
        used.add(b["frm"])
        used.add(b["to"])
    unused = [s for s in req.stations if s not in used]
    if unused:
        raise AdjustmentError(
            "待求站点没有任何观测连接，必然秩亏: " + ", ".join(unused),
            "unobserved_station", {"stations": unused},
        )

    h_stations = [
        s for s in req.stations
        if any(s in (r["frm"], r["to"]) and r["dh"] is not None for r in obs)
    ]
    bl_stations = [
        s for s in req.stations
        if any(s in (b["frm"], b["to"]) for b in bls)
    ]
    for s in bl_stations:
        if s not in h_stations:
            h_stations.append(s)
    return {
        "known": known,
        "horiz_active": horiz_active,
        "height_active": height_active,
        "height_stations": h_stations,
        "baseline_stations": set(bl_stations),
        "warnings": warnings,
        "h_adj": h_adj,
        "v_adj": v_adj,
    }


# ---------------------------------------------------------------------------
# 2. 初值（从已知点沿完整边双向 BFS 传播）
# ---------------------------------------------------------------------------

def _initial_coords(req, obs, bls, info):
    x0 = {}
    if info["horiz_active"]:
        filled = {n: np.array([p.x, p.y], dtype=float) for n, p in info["known"].items()}
        # 完整边（方位角+距离）按几何关系双向入表：已知任一端即可推算另一端
        complete = defaultdict(list)
        for r in obs:
            if r["az"] is not None and r["dist"] is not None:
                d = r["dist"] * np.array([math.cos(r["az"]), math.sin(r["az"])])
                complete[r["frm"]].append((r["to"], d))
                complete[r["to"]].append((r["frm"], -d))
        # GNSS 基线直接给出平面向量，同样可双向传播
        for b in bls:
            d = b["vec"][:2].copy()
            complete[b["frm"]].append((b["to"], d))
            complete[b["to"]].append((b["frm"], -d))
        progress = True
        while progress:
            progress = False
            for u, edges in list(complete.items()):
                if u not in filled:
                    continue
                base = filled[u]
                for v, d in edges:
                    if v in filled:
                        continue
                    filled[v] = base + d
                    progress = True
        missing = sorted({
            s for s in req.stations
            if s not in filled and (
                any(r["az"] is not None or r["dist"] is not None
                    for r in obs if s in (r["frm"], r["to"]))
                or any(s in (b["frm"], b["to"]) for b in bls))
        })
        if missing:
            raise AdjustmentError(
                "无法由已知点推算初值（需要与已知点连通、方位角与距离齐全的"
                "导线边或 GNSS 基线，边的方向不限）: " + ", ".join(missing),
                "no_initial_coordinates", {"stations": missing},
            )
        for s in req.stations:
            if s in filled:
                x0[s] = {"x": float(filled[s][0]), "y": float(filled[s][1])}

    h0 = {}
    if info["height_active"]:
        hfilled = {p.name: p.h for p in req.known if p.h is not None}
        dh_edges = defaultdict(list)
        for r in obs:
            if r["dh"] is not None:
                dh_edges[r["frm"]].append((r["to"], r["dh"]))
                dh_edges[r["to"]].append((r["frm"], -r["dh"]))
        for b in bls:
            d = float(b["vec"][2])
            dh_edges[b["frm"]].append((b["to"], d))
            dh_edges[b["to"]].append((b["frm"], -d))
        q = deque(list(hfilled.keys()))
        while q:
            u = q.popleft()
            for v, dh in dh_edges[u]:
                if v not in hfilled:
                    hfilled[v] = hfilled[u] + dh
                    q.append(v)
        for s in info["height_stations"]:
            h0[s] = float(hfilled[s])
    return x0, h0


# ---------------------------------------------------------------------------
# 3. 迭代加权最小二乘（标量观测 + Cholesky 白化的 GNSS 基线）
# ---------------------------------------------------------------------------

def _build_rows(obs):
    rows = []
    for r in obs:
        if r["az"] is not None:
            rows.append((r["id"], "az", r["az"], r["std_az"], r["weight"]))
        if r["dist"] is not None:
            rows.append((r["id"], "dist", r["dist"], r["std_dist"], r["weight"]))
        if r["dh"] is not None:
            rows.append((r["id"], "dh", r["dh"], r["std_dh"], r["weight"]))
    return rows


def _baseline_model(frm, to, coords, heights, pidx):
    """基线计算向量 (Δx,Δy,Δh) 与按参数列的 3×n 雅可比（只填待求参数）。"""
    xf = coords[frm][0] if frm in coords else None
    yf = coords[frm][1] if frm in coords else None
    xt = coords[to][0] if to in coords else None
    yt = coords[to][1] if to in coords else None
    hf = heights.get(frm)
    ht = heights.get(to)
    val = np.array([xt - xf, yt - yf, ht - hf], dtype=float)
    Jd = {}
    for end, sgn in ((to, 1.0), (frm, -1.0)):
        if end not in pidx:
            continue
        if "x" in pidx[end]:
            Jd[pidx[end]["x"]] = np.array([sgn, 0.0, 0.0])
            Jd[pidx[end]["y"]] = np.array([0.0, sgn, 0.0])
        if "h" in pidx[end]:
            Jd[pidx[end]["h"]] = np.array([0.0, 0.0, sgn])
    return val, Jd


def _model_and_jacobian(kind, frm, to, coords, heights, pidx):
    """返回 (计算值, {param_index: 偏导})；方位角计算值已归一化。"""
    if kind == "dh":
        val = heights[to] - heights[frm]
        deriv = {}
        if to in pidx and "h" in pidx[to]:
            deriv[pidx[to]["h"]] = 1.0
        if frm in pidx and "h" in pidx[frm]:
            deriv[pidx[frm]["h"]] = -1.0
        return val, deriv

    xf, yf = coords[frm]
    xt, yt = coords[to]
    dx, dy = xt - xf, yt - yf
    s2 = dx * dx + dy * dy
    s = math.sqrt(s2)
    if s < 1e-12:
        raise AdjustmentError(
            f"{frm} 与 {to} 坐标重合，无法建立 {kind} 观测方程", "coincident_points"
        )
    if kind == "dist":
        val = s
        deriv = {}
        if to in pidx:
            deriv[pidx[to]["x"]] = dx / s
            deriv[pidx[to]["y"]] = dy / s
        if frm in pidx:
            deriv[pidx[frm]["x"]] = -dx / s
            deriv[pidx[frm]["y"]] = -dy / s
        return val, deriv

    val = normalize_angle(math.atan2(dy, dx))
    deriv = {}
    if to in pidx:
        deriv[pidx[to]["x"]] = -dy / s2
        deriv[pidx[to]["y"]] = dx / s2
    if frm in pidx:
        deriv[pidx[frm]["x"]] = dy / s2
        deriv[pidx[frm]["y"]] = -dx / s2
    return val, deriv


def _rank_deficiency_details(N, names, rank):
    svals, Vh = np.linalg.svd(N)
    null_vecs = Vh[rank:].conj().T
    dims, desc = [], []
    for col in range(null_vecs.shape[1]):
        v = np.abs(np.asarray(null_vecs)[:, col])
        top = np.argsort(v)[-4:][::-1]
        items = [f"{names[i][0]}.{names[i][1]}" for i in top if v[i] > 0.1]
        dims.append(items)
        desc.append("、".join(items) if items else "未知维度")
    return {
        "rank": rank,
        "parameter_count": len(names),
        "deficiency": len(names) - rank,
        "null_space_dims": dims,
        "description": "；".join(desc),
        "singular_values": [float(x) for x in svals],
    }


def _solve_adjustment(req, obs, bls, info, x0, h0):
    rows = _build_rows(obs)
    obs_by_id = {r["id"]: r for r in obs}
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
    n = len(names)
    m = len(rows) + 3 * len(bls)   # 白化后的总观测方程行数

    coords = {k: [p.x, p.y] for k, p in info["known"].items()}
    for st, v in x0.items():
        coords[st] = [v["x"], v["y"]]
    heights = {p.name: p.h for p in req.known if p.h is not None}
    heights.update(h0)

    params = np.zeros(n)
    for st, comps in pidx.items():
        for comp, idx in comps.items():
            params[idx] = coords[st][0] if comp == "x" else (
                coords[st][1] if comp == "y" else heights[st])

    # 白化后的总雅可比/闭合差/权（基线 Cholesky 白化后按单位权）
    A = np.zeros((m, n))
    W = np.zeros(m)
    l_vec = np.zeros(m)
    # 原始（物理单位）残差与雅可比，按组保存供精度统计
    scalar_groups = [
        {"id": oid, "kind": kind, "observed": observed, "std": std,
         "weight": wf, "row": i,
         "frm": obs_by_id[oid]["frm"], "to": obs_by_id[oid]["to"]}
        for i, (oid, kind, observed, std, wf) in enumerate(rows)
    ]
    baseline_groups = [
        {"id": b["id"], "frm": b["frm"], "to": b["to"], "observed": b["vec"],
         "C": b["C"], "Linv": b["Linv"], "weight": b["weight"],
         "unit_factor": b["unit_factor"],
         "row": len(rows) + 3 * k, "blk": None, "J": None}
        for k, b in enumerate(bls)
    ]
    convergence = {"iterations": 0, "converged": False, "max_dx": None}
    rank = n
    svals = np.zeros(n)

    for it in range(1, req.max_iterations + 1):
        A.fill(0.0)
        # 标量观测
        for g in scalar_groups:
            i = g["row"]
            computed, deriv = _model_and_jacobian(
                g["kind"], g["frm"], g["to"], coords, heights, pidx)
            l_vec[i] = angle_diff(g["observed"] - computed) if g["kind"] == "az" \
                else g["observed"] - computed
            W[i] = g["weight"] / (g["std"] * g["std"])
            for j, d in deriv.items():
                A[i, j] = d
        # GNSS 基线：l = L⁻¹·(观测-计算)，B = L⁻¹·J，W = w·I
        for g in baseline_groups:
            i = g["row"]
            computed, Jd = _baseline_model(
                g["frm"], g["to"], coords, heights, pidx)
            mis = g["observed"] - computed
            Jphys = np.zeros((3, n))
            for j, col in Jd.items():
                Jphys[:, j] = col
            Linv = g["Linv"]
            wf = g["weight"]
            scale = math.sqrt(wf)
            lw = Linv @ mis * scale
            Bw = Linv @ Jphys * scale
            A[i:i + 3, :] = Bw
            l_vec[i:i + 3] = lw
            W[i:i + 3] = 1.0
            g["blk"] = mis      # 末次迭代的物理闭合差（迭代结束后即残差反号）
            g["J"] = Jphys

        N = A.T @ (W[:, None] * A)
        u = A.T @ (W * l_vec)

        svals = np.linalg.svd(N, compute_uv=False)
        smax = float(svals[0]) if len(svals) else 0.0
        tol = max(m, n) * np.finfo(float).eps * smax if smax > 0 else 0.0
        rank = int(np.sum(svals > tol))
        if rank < n:
            d = _rank_deficiency_details(N, names, rank)
            raise AdjustmentError(
                "法方程秩亏：图形/基准不足以确定全部参数（缺秩 "
                f"{n - rank} 维）：{d['description']}",
                "rank_deficient", d,
            )

        dx = np.linalg.solve(N, u)
        max_dx = float(np.max(np.abs(dx))) if n else 0.0
        params = params + dx
        for st, comps in pidx.items():
            for comp, idx in comps.items():
                if comp == "x":
                    coords[st][0] = params[idx]
                elif comp == "y":
                    coords[st][1] = params[idx]
                else:
                    heights[st] = params[idx]
        convergence.update(iterations=it, max_dx=max_dx)
        if max_dx < req.convergence_tol:
            convergence["converged"] = True
            break

    if not convergence["converged"]:
        raise AdjustmentError(
            f"平差迭代 {req.max_iterations} 次后仍未收敛（末次最大改正 "
            f"{convergence['max_dx']:.3e} m）",
            "not_converged", convergence,
        )

    return {
        "rows": rows, "scalar_groups": scalar_groups,
        "baseline_groups": baseline_groups,
        "pidx": pidx, "names": names, "A": A, "W": W, "l": l_vec,
        "N": N, "coords": coords, "heights": heights, "params": params,
        "m": m, "n": n, "dof": m - n, "rank": rank,
        "singular_values": svals, "convergence": convergence,
    }


# ---------------------------------------------------------------------------
# 4. 残差、精度统计、粗差（标量观测 + GNSS 基线组）
# ---------------------------------------------------------------------------

def _sym_eigclip(M):
    """对称阵特征值截断（负特征值视为 0），返回 (特征值, 特征向量)。"""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.clip(w, 0.0, None)
    return w, V


def _residuals_and_stats(req, obs, bls, sol):
    A, W, l_vec = sol["A"], sol["W"], sol["l"]
    m, n, dof = sol["m"], sol["n"], sol["dof"]

    v = -l_vec                       # 白化空间改正数
    chi2 = float(v @ (W * v))
    sigma0 = math.sqrt(chi2 / dof) if dof > 0 else 1.0

    Ninv = np.linalg.inv(sol["N"])

    # 标量观测的逐行统计
    WA = W[:, None] * A
    leverage = np.einsum("ij,jk,ik->i", WA, Ninv, A)
    leverage = np.clip(leverage, 0.0, 1.0)
    qv_diag = (1.0 / W) * (1.0 - leverage)
    qv_diag = np.where(qv_diag > 1e-14, qv_diag, np.nan)

    grouped: dict[str, dict] = {}
    outlier_rows = []
    max_abs_std = max_abs_w = None
    for g in sol["scalar_groups"]:
        i = g["row"]
        oid, kind = g["id"], g["kind"]
        r = next(rr for rr in obs if rr["id"] == oid)
        vi = float(v[i])
        if not math.isnan(qv_diag[i]):
            se_post = sigma0 * math.sqrt(qv_diag[i])
            se_pri = math.sqrt(qv_diag[i])
        else:
            se_post = se_pri = float("nan")
        std_res = float(vi / se_post) if se_post > 0 and not math.isnan(se_post) else None
        w_test = float(vi / se_pri) if se_pri > 0 and not math.isnan(se_pri) else None
        # 粗差判别用先验 Baarda w 检验（后验 t 在 σ0 极小时会被舍入残差放大）
        is_out = bool(w_test is not None and abs(w_test) >= req.outlier_threshold)
        comp = {
            "type": {"az": "azimuth", "dist": "distance", "dh": "height_difference"}[kind],
            "residual_m": None if kind == "az" else vi,
            "residual_rad": vi if kind == "az" else None,
            "residual_arcsec": rad_to_deg(vi) * 3600.0 if kind == "az" else None,
            "residual_in_input_unit": _residual_in_unit(kind, vi, req),
            # 后验标准化残差 t = v / (σ0·σv)；先验 Baarda w = v / σv
            "standardized_residual": std_res,
            "normalized_residual_prior": w_test,
            "standard_error_m": None if kind == "az" else (
                None if math.isnan(se_post) else se_post),
            "standard_error_arcsec": (
                rad_to_deg(se_post) * 3600.0
                if kind == "az" and not math.isnan(se_post) else None),
            "leverage": float(leverage[i]),
            "is_outlier": is_out,
        }
        hint = _outlier_hint(kind, vi,
                             w_test if w_test is not None else std_res,
                             req.outlier_threshold)
        if hint:
            comp["hint"] = hint
        gdict = grouped.setdefault(oid, {
            "id": oid, "from": r["frm"], "to": r["to"],
            "components": {}, "suspect": False, "hints": [],
        })
        key = {"az": "azimuth", "dist": "distance", "dh": "height_difference"}[kind]
        gdict["components"][key] = comp
        if is_out:
            gdict["suspect"] = True
            gdict["hints"].append(f"{key}: {hint}" if hint else f"{key}: 残差超限")
            outlier_rows.append((oid, key, std_res, w_test))
        if std_res is not None:
            max_abs_std = abs(std_res) if max_abs_std is None else max(max_abs_std, abs(std_res))
        if w_test is not None:
            max_abs_w = abs(w_test) if max_abs_w is None else max(max_abs_w, abs(w_test))

    observation_results = [grouped[r["id"]] for r in obs]

    # ---- GNSS 基线：物理空间的相关残差 / Mahalanobis / 协方差贡献 ----
    baseline_results = []
    pidx = sol["pidx"]
    for g in sol["baseline_groups"]:
        i3 = slice(g["row"], g["row"] + 3)
        wf = g["weight"]
        C0 = g["C"]                          # 接收机给出的先验协方差
        C = C0 / wf                          # 计入额外权重因子后的有效先验协方差
        C0inv = np.linalg.inv(C0)
        J = g["J"]
        vphys = -g["blk"].astype(float)     # 物理改正数 = 计算值 - 观测值
        Bw = A[i3, :]
        # 白化空间杠杆矩阵 H = B N⁻¹ Bᵀ（对称幂等子块）
        H = Bw @ Ninv @ Bw.T
        lev_trace = float(np.trace(H))
        R = np.eye(3) - H                   # 冗余因子矩阵（白化空间）
        eig_r = np.linalg.eigvalsh(0.5 * (R + R.T))
        r_min = float(max(eig_r.min(), 0.0))
        redundancy = float(np.clip(np.trace(R) / 3.0, 0.0, 1.0))

        # 物理残差协方差：白化空间 l_w = √w·L⁻¹·l，v_phys = (1/√w)·L·v_w，
        # 故 Cv = (1/w)·L·(I−H)·Lᵀ = C₀/w − J·N⁻¹·Jᵀ
        # （注意：右端折算回物理空间时 w 恰好约去，不能再乘 wf）
        Cv_prior = C - J @ Ninv @ J.T
        Cv_prior = 0.5 * (Cv_prior + Cv_prior.T)
        # 完整 3×3 矩阵求逆（保留 dx/dy/dh 间全部相关性）；
        # 秩亏（如多余观测为 0）时按特征值容差做 Moore-Penrose 伪逆
        wv, Vv = np.linalg.eigh(Cv_prior)
        tol = 1e-11 * max(1.0, float(np.max(np.abs(wv))))
        pos = wv > tol
        rank_cv = int(np.sum(pos))
        Cv_post = sigma0**2 * Cv_prior

        # 先验 Mahalanobis：w² = vᵀ Cv⁻¹ v（Baarda 数据探测，相关观测整体检验）
        if rank_cv > 0:
            inv_prior = (Vv[:, pos] * (1.0 / wv[pos])) @ Vv[:, pos].T
            maha_prior = float(max(vphys @ inv_prior @ vphys, 0.0))
        else:
            maha_prior = None
        maha_post = (
            maha_prior / sigma0**2
            if maha_prior is not None and sigma0 > 0 else None)
        p_prior = (
            float(stats.chi2.sf(maha_prior, rank_cv))
            if maha_prior is not None else None)

        # 分量级先验 Baarda w：必须取物理坐标（dx,dy,dh）顺序下 Cv 的对角元，
        # 不能使用排序后的特征值
        comp_w, comp_out = [], False
        for k, cname in enumerate(("dx", "dy", "dh")):
            var_k = float(Cv_prior[k, k])
            if var_k > tol:
                se = math.sqrt(max(var_k, 0.0))
                wk = float(vphys[k] / se)
            else:
                se, wk = 0.0, None
            isc = bool(wk is not None and abs(wk) >= req.outlier_threshold)
            comp_out = comp_out or isc
            comp_w.append({
                "component": cname,
                "w": wk,
                "residual_standard_error_m": se if wk is not None else None,
                "is_outlier": isc,
            })
            if wk is not None:
                max_abs_w = abs(wk) if max_abs_w is None else max(max_abs_w, abs(wk))
        # 整体粗差：Mahalanobis 超过 df=rank_cv 的临界值，或任一分量 w 超限
        if rank_cv > 0 and maha_prior is not None:
            maha_crit = float(stats.chi2.ppf(
                1.0 - 2.0 * (1.0 - stats.norm.cdf(req.outlier_threshold)),
                rank_cv))
            group_out = bool(maha_prior >= maha_crit or comp_out)
        else:
            group_out = bool(comp_out)

        # 协方差贡献（信息矩阵 w·C₀⁻¹：基线对法方程的精度贡献）
        precision = wf * C0inv
        # 端点后验协方差缩减：N⁻¹ - (N - w·JᵀC₀⁻¹J)⁻¹ 在端点参数上的子块
        Nb = wf * J.T @ C0inv @ J
        endpoints = []
        free = [
            (end, [pidx[end][c] for c in ("x", "y", "h") if c in pidx.get(end, {})])
            for end in (g["frm"], g["to"])
        ]
        free = [(e, idx) for e, idx in free if idx]
        try:
            N_other = sol["N"] - Nb
            inv_other = np.linalg.inv(N_other)
        except np.linalg.LinAlgError:
            inv_other = None
        for k, (end, idx) in enumerate(free):
            role = "from" if end == g["frm"] else "to"
            if inv_other is None:
                endpoints.append({
                    "name": end, "role": role,
                    "covariance_reduction_m2": None,
                    "trace_reduction_m2": None,
                    "positive_semidefinite": None,
                })
                continue
            d3 = inv_other[np.ix_(idx, idx)] - Ninv[np.ix_(idx, idx)]
            wd, _ = _sym_eigclip(d3)
            endpoints.append({
                "name": end, "role": role,
                "covariance_reduction_m2": [
                    [float(x) for x in row] for row in d3],
                "trace_reduction_m2": float(np.trace(d3)),
                "positive_semidefinite": bool(np.all(wd >= -1e-10)),
            })

        hint = None
        if group_out:
            hint = "基线向量整体或某分量与其余网形不一致，请核对天线高、" \
                   "起终点方向与时段观测质量（含相关性的 Mahalanobis 检验）"

        unit_factor = g["unit_factor"]
        result = {
            "id": g["id"], "from": g["frm"], "to": g["to"],
            "residual_vector_m": [float(x) for x in vphys],
            "residual_vector": {
                c: float(vphys[k]) / unit_factor
                for k, c in enumerate(("dx", "dy", "dh"))},
            "residual_norm_m": float(np.linalg.norm(vphys)),
            "residual_covariance_m2": [
                [float(x) for x in row] for row in Cv_prior],
            "residual_covariance_posterior_m2": [
                [float(x) for x in row] for row in Cv_post],
            "mahalanobis_prior": maha_prior,
            "mahalanobis_posterior": maha_post,
            "chi2_df": rank_cv if maha_prior is not None else None,
            "p_value": p_prior,
            "component_tests": comp_w,
            "leverage": lev_trace,
            "redundancy_number": redundancy,
            "min_redundancy_eigenvalue": r_min,
            "precision_contribution_m2": [
                [float(x) for x in row] for row in precision],
            "normal_matrix_contribution": {
                "trace_m2": float(np.trace(Nb)),
                "rank": int(np.linalg.matrix_rank(Nb)),
            },
            "endpoint_covariance_contribution": endpoints,
            "is_outlier": group_out,
            "suspect": group_out,
            "hint": hint,
        }
        baseline_results.append(result)
        if group_out:
            outlier_rows.append((g["id"], "gnss_baseline", None, math.sqrt(maha_prior)))

    p_value = float(1.0 - stats.chi2.cdf(chi2, dof)) if dof > 0 else None
    crit = (
        [float(stats.chi2.ppf(0.025, dof)), float(stats.chi2.ppf(0.975, dof))]
        if dof > 0 else None
    )
    return {
        "observation_results": observation_results,
        "baseline_results": baseline_results,
        "outlier_rows": outlier_rows,
        "sigma0": sigma0,
        "Ninv": Ninv,
        "statistics": {
            "observations_m": m,
            "scalar_rows": len(sol["scalar_groups"]),
            "baselines": len(sol["baseline_groups"]),
            "baseline_rows": 3 * len(sol["baseline_groups"]),
            "parameters_n": n,
            "rank": int(sol["rank"]),
            "rank_deficiency": int(n - sol["rank"]),
            "degrees_of_freedom": dof,
            "weighted_ssr": chi2,
            "sigma0_posterior": sigma0,
            "sigma0_apriori": 1.0,
            "chi2_p_value": p_value,
            "chi2_critical_025_975": crit,
            "max_abs_standardized_residual": max_abs_std,
            "max_abs_normalized_residual_baarda": max_abs_w,
        },
    }


def _residual_in_unit(kind, v, req):
    if kind == "az":
        u = req.units.angle
        if u in ("degree", "deg"):
            return rad_to_deg(v)
        if u in ("gon", "grad"):
            return rad_to_gon(v)
        if u == "dms":
            return rad_to_dms(v)["text"]
        return v
    unit = req.units.distance if kind == "dist" else req.units.height
    return m_to_distance(v, unit)


def _outlier_hint(kind, residual, std_res, threshold):
    if std_res is None or abs(std_res) < threshold:
        return None
    if kind == "az":
        d = abs(angle_diff(residual))
        if d > math.radians(170.0):
            return "残差约 ±180°，疑似前后视写反（测站与照准点颠倒）"
        if math.radians(1.0) < d < math.radians(20.0):
            return "疑似角度单位混用（degree/gon/rad）或照准方向错误"
        return "请核对方向角读数与照准目标"
    if kind == "dh":
        return "请核对高差符号（前后视写反会使高差反号）与读数"
    return "请核对距离读数与单位（m/km）"


# ---------------------------------------------------------------------------
# 5. 协方差与误差椭圆
# ---------------------------------------------------------------------------

def _point_precision(req, sol, st_res, x0, h0):
    pidx, Ninv, sigma0 = sol["pidx"], st_res["Ninv"], st_res["sigma0"]
    k_chi = float(stats.chi2.ppf(req.confidence, df=2))
    k_chi3 = float(stats.chi2.ppf(req.confidence, df=3))
    cov_unit = req.units.covariance
    u = {"m": 1.0, "km": 1000.0, "ft": 0.3048}[cov_unit]

    out = []
    for st in req.stations:
        has_xy = "x" in pidx.get(st, {})
        has_h = "h" in pidx.get(st, {})
        e: dict[str, Any] = {
            "name": st,
            "x": sol["coords"][st][0] if has_xy or st in sol["coords"] else None,
            "y": sol["coords"][st][1] if has_xy or st in sol["coords"] else None,
            "h": sol["heights"].get(st),
        }
        if st in x0:
            e["initial_x"] = x0[st]["x"]
            e["initial_y"] = x0[st]["y"]
        if st in h0:
            e["initial_h"] = h0[st]
        if has_xy:
            ix, iy = pidx[st]["x"], pidx[st]["y"]
            D = sigma0**2 * Ninv[np.ix_([ix, iy], [ix, iy])]
            cxx, cyy, cxy = D[0, 0], D[1, 1], D[0, 1]
            e["std_x_m"] = math.sqrt(max(cxx, 0.0))
            e["std_y_m"] = math.sqrt(max(cyy, 0.0))
            tr = cxx + cyy
            root = math.sqrt(max(0.25 * (cxx - cyy) ** 2 + cxy * cxy, 0.0))
            lam1 = max(0.5 * tr + root, 0.0)
            lam2 = max(0.5 * tr - root, 0.0)
            a, b = math.sqrt(lam1), math.sqrt(lam2)
            theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy) % math.pi
            e["ellipse"] = {
                "semi_major_m": a * math.sqrt(k_chi),
                "semi_minor_m": b * math.sqrt(k_chi),
                "semi_major": a * math.sqrt(k_chi) / u,
                "semi_minor": b * math.sqrt(k_chi) / u,
                "std_semi_major_m": a,
                "std_semi_minor_m": b,
                "std_semi_major": a / u,
                "std_semi_minor": b / u,
                "major_azimuth_deg": rad_to_deg(theta),
                "confidence": req.confidence,
                "chi2_scale_df2": k_chi,
                "unit": cov_unit,
            }
            e["covariance_xy"] = [
                [cxx / u**2, cxy / u**2],
                [cxy / u**2, cyy / u**2],
            ]
            e["covariance_unit"] = cov_unit
        else:
            e["ellipse"] = None
            e["covariance_xy"] = None
            e["covariance_unit"] = None
        if has_h:
            ih = pidx[st]["h"]
            e["std_h_m"] = sigma0 * math.sqrt(max(Ninv[ih, ih], 0.0))
        if has_xy and has_h:
            # GNSS 基线使 x/y/h 联合估计：输出完整 3×3 后验协方差与 3D 误差椭球
            idx = [pidx[st]["x"], pidx[st]["y"], pidx[st]["h"]]
            D3 = sigma0**2 * Ninv[np.ix_(idx, idx)]
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
                "semi_axes": [a3 / u for a3 in axes3],
                "major_axis_azimuth_deg": float(
                    rad_to_deg(math.atan2(major[1], major[0])) % 360.0),
                "major_axis_dip_deg": float(
                    rad_to_deg(math.asin(float(np.clip(major[2], -1.0, 1.0))))),
                "confidence": req.confidence,
                "chi2_scale_df3": k_chi3,
                "unit": cov_unit,
            }
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# 6. 闭合差（BFS 生成树基本环）
# ---------------------------------------------------------------------------

def _spanning_forest(edges):
    adj = defaultdict(list)
    nodes = set()
    rec_dir = {}
    for key, frm, to in edges:
        adj[frm].append((to, key))
        adj[to].append((frm, key))
        rec_dir[key] = (frm, to)
        nodes.add(frm); nodes.add(to)

    parent: dict[str, tuple] = {}
    tree_edges: set[str] = set()
    for root in sorted(nodes):
        if root in parent:
            continue
        parent[root] = (None, None, True)
        q = deque([root])
        while q:
            a = q.popleft()
            for v, key in adj[a]:
                if v in parent:
                    continue
                frm, to = rec_dir[key]
                # 从 v 走向父 a 时，是否沿记录方向 frm→to
                forward = (v == frm and a == to)
                parent[v] = (a, key, forward)
                tree_edges.add(key)
                q.append(v)
    return parent, tree_edges


def _fundamental_cycles(edges):
    """返回有向基本环列表；每项为 [(edge_key, traverse_forward), ...]。

    先沿非树边记录方向 frm→to，再沿生成树由 to 回到 frm。
    traverse_forward=True 表示按记录的 frm→to 方向通过该边。
    """
    parent, tree_edges = _spanning_forest(edges)

    def root_and_hops(node):
        hops = []
        while parent[node][0] is not None:
            up, key, forward = parent[node]
            hops.append((up, key, forward))
            node = up
        return node, hops

    cycles = []
    for key, frm, to in edges:
        if key in tree_edges:
            continue
        root_f, hops_f = root_and_hops(frm)
        root_t, hops_t = root_and_hops(to)
        if root_f != root_t:
            continue
        # frm 各祖先 -> 深度
        depth = {frm: 0}
        node = frm
        for up, _, _ in hops_f:
            depth[up] = depth[node] + 1
            node = up
        # to 上行到 LCA
        up_part = []
        node = to
        while node not in depth:
            up, k, forward = parent[node]
            up_part.append((k, forward))   # node→up 的通过方向
            node = up
        lca = node
        # LCA 下行到 frm：取 hops_f 前 depth[lca] 跳并反转
        down_part = []
        for up, k, forward in reversed(hops_f[:depth[lca]]):
            down_part.append((k, not forward))
        cycles.append([(key, True)] + up_part + down_part)
    return cycles


def _loop_closures(obs, sol):
    by_key = {r["id"]: r for r in obs}
    closures = []

    h_edges = [(r["id"], r["frm"], r["to"]) for r in obs
               if r["az"] is not None and r["dist"] is not None]
    for seq in _fundamental_cycles(h_edges):
        sx0 = sy0 = sx1 = sy1 = per = 0.0
        for key, forward in seq:
            r = by_key[key]
            a0, s0 = r["az"], r["dist"]
            ddx = sol["coords"][r["to"]][0] - sol["coords"][r["frm"]][0]
            ddy = sol["coords"][r["to"]][1] - sol["coords"][r["frm"]][1]
            a1 = normalize_angle(math.atan2(ddy, ddx))
            s1 = math.hypot(ddx, ddy)
            if not forward:
                a0 = normalize_angle(a0 + math.pi)
                a1 = normalize_angle(a1 + math.pi)
            sx0 += s0 * math.cos(a0); sy0 += s0 * math.sin(a0)
            sx1 += s1 * math.cos(a1); sy1 += s1 * math.sin(a1)
            per += s0
        pre, post = math.hypot(sx0, sy0), math.hypot(sx1, sy1)
        closures.append({
            "type": "horizontal",
            "edges": [k for k, _ in seq],
            "perimeter_m": per,
            "pre_misclosure_m": pre,
            "post_misclosure_m": post,
            "pre_relative": pre / per if per > 0 else None,
            "post_relative": post / per if per > 0 else None,
            "pre_components_m": [sx0, sy0],
            "post_components_m": [sx1, sy1],
        })

    v_edges = [(r["id"], r["frm"], r["to"]) for r in obs if r["dh"] is not None]
    for seq in _fundamental_cycles(v_edges):
        pre = post = length = 0.0
        for key, forward in seq:
            r = by_key[key]
            d0 = r["dh"]
            d1 = sol["heights"][r["to"]] - sol["heights"][r["frm"]]
            if not forward:
                d0, d1 = -d0, -d1
            pre += d0; post += d1
            length += r["dist"] or 0.0
        closures.append({
            "type": "height",
            "edges": [k for k, _ in seq],
            "route_length_m": length,
            "pre_misclosure_m": pre,
            "post_misclosure_m": post,
        })
    return closures[:100]


# ---------------------------------------------------------------------------
# 7. 主入口
# ---------------------------------------------------------------------------

def run_adjustment(req: AdjustmentRequest) -> dict[str, Any]:
    """执行完整预检 + 平差 + 统计，返回结果字典（含 _internal 供预演比对）。"""
    if not req.observations and not req.baselines:
        raise AdjustmentError(
            "请求未包含任何观测（observations）或 GNSS 基线（baselines）",
            "empty_observation",
        )
    obs, dup_obs = _normalized_observations(req)
    bls, dup_bls = _normalized_baselines(req)
    info = _validate_network(req, obs, bls)
    x0, h0 = _initial_coords(req, obs, bls, info)
    sol = _solve_adjustment(req, obs, bls, info, x0, h0)
    st_res = _residuals_and_stats(req, obs, bls, sol)
    stations_out = _point_precision(req, sol, st_res, x0, h0)
    closures = _loop_closures(obs, sol)

    warnings = list(info["warnings"])
    duplicates = dup_obs + dup_bls
    if duplicates:
        warnings.append(
            "发现重复观测（同测站/目标/类型），已作为独立观测按权参与平差："
            + "; ".join(f"{d['key']}: {d['ids']}" for d in duplicates)
        )
    if sol["dof"] == 0:
        warnings.append("多余观测数为 0：无校核条件，无法进行粗差判别与精度评定")
    if bls:
        warnings.append(
            f"含 {len(bls)} 条 GNSS 三维基线：协方差阵经 Cholesky 分解白化残差"
            "与雅可比后联合解算，粗差按含相关性的 Mahalanobis（Baarda）检验判别"
        )

    return {
        "status": "ok",
        "name": req.name,
        "units": req.units.model_dump(),
        "accuracy": req.accuracy.model_dump(),
        "outlier_threshold": req.outlier_threshold,
        "confidence": req.confidence,
        "stations": stations_out,
        "known_points": [p.model_dump() for p in req.known],
        "observations": st_res["observation_results"],
        "baselines": st_res["baseline_results"],
        "suspects": [
            {
                "observation_id": oid,
                "component": key,
                "kind": "gnss_baseline" if key == "gnss_baseline" else "total_station",
                "standardized_residual": sr,
                "normalized_residual_baarda": w,
            }
            for oid, key, sr, w in st_res["outlier_rows"]
        ],
        "statistics": st_res["statistics"] | {
            "iterations": sol["convergence"]["iterations"],
            "converged": sol["convergence"]["converged"],
            "last_max_correction_m": sol["convergence"]["max_dx"],
        },
        "loop_closures": closures,
        "duplicates": duplicates,
        "warnings": warnings,
        "_internal": {"obs": obs, "bls": bls, "sol": sol, "initial": (x0, h0)},
    }


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if not k.startswith("_")}
