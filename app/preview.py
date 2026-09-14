"""预演（what-if）：临时剔除观测 / 调整权重 / 停用 GNSS 基线 /
整体或逐条缩放基线协方差，与原案比较，不落库。"""
from __future__ import annotations

from .adjustment import public_result, run_adjustment
from .models import PreviewRequest


def run_preview(req: PreviewRequest) -> dict:
    base_req = req.base.model_copy(deep=True)
    base_req.save = False
    base = run_adjustment(base_req)

    # 构造预演请求
    sim = req.base.model_copy(deep=True)
    sim.save = False
    disable = set(req.disable)

    def eff_id(o, i):
        return o.id or f"o{i}"

    used_ids = {eff_id(o, i) for i, o in enumerate(base_req.observations, start=1)}
    kept = []
    for i, o in enumerate(sim.observations, start=1):
        oid = eff_id(o, i)
        if oid in disable or req.weight_overrides.get(oid, 1.0) == 0.0:
            continue
        if oid in req.weight_overrides:
            o.weight = float(req.weight_overrides[oid])
        if oid in req.std_overrides:
            factor = float(req.std_overrides[oid])
            if o.std_azimuth is not None:
                o.std_azimuth = o.std_azimuth * factor
            if o.std_distance is not None:
                o.std_distance = o.std_distance * factor
            if o.std_dh is not None:
                o.std_dh = o.std_dh * factor
        kept.append(o)

    unknown = [i for i in disable if i not in used_ids]
    unknown += [i for i in req.weight_overrides if i not in used_ids]
    unknown += [i for i in req.std_overrides if i not in used_ids]
    sim.observations = kept

    # ---- GNSS 基线：停用 / 调权 / 协方差缩放（逐条 + 整体） ----
    def eff_bid(b, i):
        return b.id or f"b{i}"

    used_bids = {eff_bid(b, i) for i, b in enumerate(base_req.baselines, start=1)}
    disable_bl = set(req.disable_baselines)
    kept_bl = []
    for i, b in enumerate(sim.baselines, start=1):
        bid = eff_bid(b, i)
        w = req.baseline_weight_overrides.get(bid, 1.0)
        if bid in disable_bl or w == 0.0:
            continue
        if bid in req.baseline_weight_overrides:
            b.weight = float(w)
        scale = 1.0
        if req.baseline_covariance_scale is not None:
            scale *= float(req.baseline_covariance_scale)
        if bid in req.baseline_covariance_scales:
            scale *= float(req.baseline_covariance_scales[bid])
        if scale != 1.0:
            b.covariance = [
                [float(c) * scale for c in row] for row in b.covariance]
        kept_bl.append(b)
    sim.baselines = kept_bl

    unknown_bl = [i for i in disable_bl if i not in used_bids]
    unknown_bl += [i for i in req.baseline_weight_overrides if i not in used_bids]
    unknown_bl += [i for i in req.baseline_covariance_scales if i not in used_bids]

    preview = run_adjustment(sim)

    base_stations = {s["name"]: s for s in base["stations"]}
    shifts = []
    for s in preview["stations"]:
        b = base_stations.get(s["name"])
        if b is None:
            continue
        dx = s["x"] - b["x"] if s["x"] is not None and b["x"] is not None else None
        dy = s["y"] - b["y"] if s["y"] is not None and b["y"] is not None else None
        dh = (s["h"] - b["h"]) if (s["h"] is not None and b["h"] is not None) else None
        shifts.append({
            "name": s["name"],
            "dx_m": dx, "dy_m": dy,
            "horizontal_shift_m": (
                (dx * dx + dy * dy) ** 0.5
                if dx is not None and dy is not None else None),
            "dh_m": dh,
            "std_x_m_base": b.get("std_x_m"), "std_x_m_preview": s.get("std_x_m"),
            "std_y_m_base": b.get("std_y_m"), "std_y_m_preview": s.get("std_y_m"),
            "std_h_m_base": b.get("std_h_m"), "std_h_m_preview": s.get("std_h_m"),
        })
    # 预演后消失（参数被整体移除，如只剩高程网）的站点也列出位移为 None
    for name, b in base_stations.items():
        if name not in {s["name"] for s in preview["stations"]}:
            shifts.append({
                "name": name, "dx_m": None, "dy_m": None,
                "horizontal_shift_m": None, "dh_m": None,
                "std_x_m_base": b.get("std_x_m"), "std_x_m_preview": None,
                "std_y_m_base": b.get("std_y_m"), "std_y_m_preview": None,
                "std_h_m_base": b.get("std_h_m"), "std_h_m_preview": None,
            })

    def closure_map(res):
        return {
            tuple(c["edges"]): c for c in res.get("loop_closures", [])
        }

    cb, cp = closure_map(base), closure_map(preview)
    closure_changes = []
    for key in cb.keys() & cp.keys():
        a, b = cb[key], cp[key]
        closure_changes.append({
            "type": a["type"], "edges": list(key),
            "pre_misclosure_m_base": a["pre_misclosure_m"],
            "post_misclosure_m_base": a["post_misclosure_m"],
            "post_misclosure_m_preview": b["post_misclosure_m"],
        })

    bstat, pstat = base["statistics"], preview["statistics"]
    return {
        "status": "ok",
        "preview": public_result(preview),
        "comparison": {
            "disabled": sorted(disable),
            "unknown_observation_ids": sorted(set(unknown)),
            "disabled_baselines": sorted(disable_bl),
            "unknown_baseline_ids": sorted(set(unknown_bl)),
            "baseline_covariance_scale": req.baseline_covariance_scale,
            "baseline_covariance_scales": req.baseline_covariance_scales,
            "baseline_count": {
                "base": bstat.get("baselines", 0),
                "preview": pstat.get("baselines", 0),
            },
            "degrees_of_freedom": {
                "base": bstat["degrees_of_freedom"],
                "preview": pstat["degrees_of_freedom"],
            },
            "rank": {
                "base": bstat.get("rank"),
                "preview": pstat.get("rank"),
            },
            "rank_deficiency": {
                "base": bstat.get("rank_deficiency"),
                "preview": pstat.get("rank_deficiency"),
            },
            "parameters": {
                "base": bstat["parameters_n"],
                "preview": pstat["parameters_n"],
            },
            "sigma0_posterior": {
                "base": bstat["sigma0_posterior"],
                "preview": pstat["sigma0_posterior"],
            },
            "chi_square": {
                "base": bstat["weighted_ssr"],
                "preview": pstat["weighted_ssr"],
            },
            "max_abs_standardized_residual": {
                "base": bstat["max_abs_standardized_residual"],
                "preview": pstat["max_abs_standardized_residual"],
            },
            "max_abs_normalized_residual_baarda": {
                "base": bstat["max_abs_normalized_residual_baarda"],
                "preview": pstat["max_abs_normalized_residual_baarda"],
            },
            "loop_closures": closure_changes,
            "station_shifts": shifts,
        },
    }
