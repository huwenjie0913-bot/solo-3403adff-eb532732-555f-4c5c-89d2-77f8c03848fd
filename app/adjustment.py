"""测量网（平面导线 + 高程）加权最小二乘平差核心。

函数模型（每条有观测的边产生若干观测方程行）：

* 方位角  a = atan2(Δy, Δx)
* 平距    s = sqrt(Δx² + Δy²)
* 高差    dh = h(to) - h(from)

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


def _validate_network(req, obs):
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

    all_names = set(known) | station_set
    for r in obs:
        for end in (r["frm"], r["to"]):
            if end not in all_names:
                raise AdjustmentError(
                    f"观测 {r['id']} 的端点 {end} 既不是已知点也不是待求站点",
                    "unknown_endpoint", {"id": r["id"], "endpoint": end},
                )

    horiz_active = any(r["az"] is not None or r["dist"] is not None for r in obs)
    height_active = any(r["dh"] is not None for r in obs)

    h_adj, v_adj = defaultdict(set), defaultdict(set)
    for r in obs:
        if r["az"] is not None or r["dist"] is not None:
            h_adj[r["frm"]].add(r["to"])
            h_adj[r["to"]].add(r["frm"])
        if r["dh"] is not None:
            v_adj[r["frm"]].add(r["to"])
            v_adj[r["to"]].add(r["frm"])

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

    if horiz_active:
        if len(known) < 2:
            raise AdjustmentError(
                "平面网基准不足：至少需要 2 个已知坐标点以固定平移与旋转"
                f"（当前 {len(known)} 个）",
                "insufficient_datum", {"known_count": len(known)},
            )
        unreachable = []
        for comp in components(h_adj, set(h_adj.keys())):
            if not any(n in known for n in comp):
                unreachable.extend(n for n in comp if n not in known)
        if unreachable:
            raise AdjustmentError(
                "平面网存在不与任何已知点连通的部分: " + ", ".join(sorted(unreachable)),
                "disconnected", {"stations": sorted(unreachable)},
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

    used = set()
    for r in obs:
        used.add(r["frm"])
        used.add(r["to"])
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
    return {
        "known": known,
        "horiz_active": horiz_active,
        "height_active": height_active,
        "height_stations": h_stations,
        "warnings": warnings,
        "h_adj": h_adj,
        "v_adj": v_adj,
    }


# ---------------------------------------------------------------------------
# 2. 初值（从已知点按完整边 BFS 传播）
# ---------------------------------------------------------------------------

def _initial_coords(req, obs, info):
    x0 = {}
    if info["horiz_active"]:
        filled = {n: np.array([p.x, p.y], dtype=float) for n, p in info["known"].items()}
        complete = defaultdict(list)
        for r in obs:
            if r["az"] is not None and r["dist"] is not None:
                complete[r["frm"]].append(r)
        progress = True
        while progress:
            progress = False
            for frm, rows in list(complete.items()):
                if frm not in filled:
                    continue
                base = filled[frm]
                for r in rows:
                    if r["to"] in filled:
                        continue
                    d = r["dist"] * np.array([math.cos(r["az"]), math.sin(r["az"])])
                    filled[r["to"]] = base + d
                    progress = True
        missing = sorted({
            s for s in req.stations
            if s not in filled and any(
                r["az"] is not None or r["dist"] is not None
                for r in obs if s in (r["frm"], r["to"]))
        })
        if missing:
            raise AdjustmentError(
                "无法由已知点推算初值（需要从已知点出发、方位角与距离齐全的导线边）: "
                + ", ".join(missing),
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
# 3. 迭代加权最小二乘
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


def _solve_adjustment(req, obs, info, x0, h0):
    rows = _build_rows(obs)
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
    n, m = len(names), len(rows)

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

    A = np.zeros((m, n))
    W = np.zeros(m)
    l_vec = np.zeros(m)
    convergence = {"iterations": 0, "converged": False, "max_dx": None}

    for it in range(1, req.max_iterations + 1):
        A.fill(0.0)
        for i, (oid, kind, observed, std, wf) in enumerate(rows):
            r = next(rr for rr in obs if rr["id"] == oid)
            computed, deriv = _model_and_jacobian(
                kind, r["frm"], r["to"], coords, heights, pidx)
            l_vec[i] = angle_diff(observed - computed) if kind == "az" \
                else observed - computed
            W[i] = wf / (std * std)
            for j, d in deriv.items():
                A[i, j] = d

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
        "rows": rows, "pidx": pidx, "names": names, "A": A, "W": W, "l": l_vec,
        "N": N, "coords": coords, "heights": heights, "params": params,
        "m": m, "n": n, "dof": m - n, "convergence": convergence,
    }


# ---------------------------------------------------------------------------
# 4. 残差、精度统计、粗差
# ---------------------------------------------------------------------------

def _residuals_and_stats(req, obs, sol):
    rows, A, W, l_vec = sol["rows"], sol["A"], sol["W"], sol["l"]
    m, n, dof = sol["m"], sol["n"], sol["dof"]

    v = -l_vec                       # 改正数 v = 模型值 - 观测值
    chi2 = float(np.sum(W * v * v))
    sigma0 = math.sqrt(chi2 / dof) if dof > 0 else 1.0

    Ninv = np.linalg.inv(sol["N"])

    # 残差协方差对角：Q_vv = Q - A N⁻¹ Aᵀ。
    # 直接整体相减在强定权边上会因舍入抵消产生负/极小方差，
    # 故用杠杆值 h_ii = (W·A)·N⁻¹·Aᵀ 逐行计算，数值稳定。
    WA = W[:, None] * A
    leverage = np.einsum("ij,jk,ik->i", WA, Ninv, A)
    leverage = np.clip(leverage, 0.0, 1.0)
    qv_diag = (1.0 / W) * (1.0 - leverage)
    # 完全固定（冗余为 0）的观测，残差恒为 0、无法标准化
    qv_diag = np.where(qv_diag > 1e-14, qv_diag, np.nan)

    grouped: dict[str, dict] = {}
    outlier_rows = []
    max_abs_std = max_abs_w = None
    for i, (oid, kind, observed, std, wf) in enumerate(rows):
        r = next(rr for rr in obs if rr["id"] == oid)
        if not math.isnan(qv_diag[i]):
            se_post = sigma0 * math.sqrt(qv_diag[i])
            se_pri = math.sqrt(qv_diag[i])
        else:
            se_post = se_pri = float("nan")
        std_res = float(v[i] / se_post) if se_post > 0 and not math.isnan(se_post) else None
        w_test = float(v[i] / se_pri) if se_pri > 0 and not math.isnan(se_pri) else None
        # 粗差判别用先验 Baarda w 检验（后验 t 在 σ0 极小时会被舍入残差放大）
        is_out = bool(w_test is not None and abs(w_test) >= req.outlier_threshold)
        comp = {
            "type": {"az": "azimuth", "dist": "distance", "dh": "height_difference"}[kind],
            "residual_m": None if kind == "az" else float(v[i]),
            "residual_rad": float(v[i]) if kind == "az" else None,
            "residual_arcsec": rad_to_deg(v[i]) * 3600.0 if kind == "az" else None,
            "residual_in_input_unit": _residual_in_unit(kind, v[i], req),
            # 后验标准化残差 t = v / (σ0·σv)；先验 Baarda w = v / σv
            "standardized_residual": std_res,
            "normalized_residual_prior": w_test,
            "standard_error_m": None if kind == "az" else (
                None if math.isnan(se_post) else float(se_post)),
            "standard_error_arcsec": (
                rad_to_deg(se_post) * 3600.0
                if kind == "az" and not math.isnan(se_post) else None),
            "leverage": float(leverage[i]),
            "is_outlier": is_out,
        }
        hint = _outlier_hint(kind, v[i],
                             w_test if w_test is not None else std_res,
                             req.outlier_threshold)
        if hint:
            comp["hint"] = hint
        g = grouped.setdefault(oid, {
            "id": oid, "from": r["frm"], "to": r["to"],
            "components": {}, "suspect": False, "hints": [],
        })
        key = {"az": "azimuth", "dist": "distance", "dh": "height_difference"}[kind]
        g["components"][key] = comp
        if is_out:
            g["suspect"] = True
            g["hints"].append(f"{key}: {hint}" if hint else f"{key}: 残差超限")
            outlier_rows.append((oid, key, std_res, w_test))
        if std_res is not None:
            max_abs_std = abs(std_res) if max_abs_std is None else max(max_abs_std, abs(std_res))
        if w_test is not None:
            max_abs_w = abs(w_test) if max_abs_w is None else max(max_abs_w, abs(w_test))

    observation_results = [grouped[r["id"]] for r in obs]
    p_value = float(1.0 - stats.chi2.cdf(chi2, dof)) if dof > 0 else None
    crit = (
        [float(stats.chi2.ppf(0.025, dof)), float(stats.chi2.ppf(0.975, dof))]
        if dof > 0 else None
    )
    return {
        "observation_results": observation_results,
        "outlier_rows": outlier_rows,
        "sigma0": sigma0,
        "Ninv": Ninv,
        "statistics": {
            "observations_m": m,
            "parameters_n": n,
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
    cov_unit = req.units.covariance
    u = {"m": 1.0, "km": 1000.0, "ft": 0.3048}[cov_unit]

    out = []
    for st in req.stations:
        has_xy = "x" in pidx.get(st, {})
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
        if "x" in pidx.get(st, {}):
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
        if "h" in pidx.get(st, {}):
            e["std_h_m"] = sigma0 * math.sqrt(max(Ninv[pidx[st]["h"], pidx[st]["h"]], 0.0))
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
    obs, duplicates = _normalized_observations(req)
    info = _validate_network(req, obs)
    x0, h0 = _initial_coords(req, obs, info)
    sol = _solve_adjustment(req, obs, info, x0, h0)
    st_res = _residuals_and_stats(req, obs, sol)
    stations_out = _point_precision(req, sol, st_res, x0, h0)
    closures = _loop_closures(obs, sol)

    warnings = list(info["warnings"])
    if duplicates:
        warnings.append(
            "发现重复观测（同测站/目标/类型），已作为独立观测按权参与平差："
            + "; ".join(f"{d['key']}: {d['ids']}" for d in duplicates)
        )
    if sol["dof"] == 0:
        warnings.append("多余观测数为 0：无校核条件，无法进行粗差判别与精度评定")

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
        "suspects": [
            {
                "observation_id": oid,
                "component": key,
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
        "_internal": {"obs": obs, "sol": sol, "initial": (x0, h0)},
    }


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if not k.startswith("_")}
