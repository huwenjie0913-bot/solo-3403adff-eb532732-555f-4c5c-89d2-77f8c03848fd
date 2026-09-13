"""预演（what-if）：临时剔除观测 / 调整权重，与原案比较，不落库。"""
from __future__ import annotations

import copy

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

    preview = run_adjustment(sim)

    base_stations = {s["name"]: s for s in base["stations"]}
    shifts = []
    for s in preview["stations"]:
        b = base_stations.get(s["name"])
        if b is None:
            continue
        dx = s["x"] - b["x"]
        dy = s["y"] - b["y"]
        dh = (s["h"] - b["h"]) if (s["h"] is not None and b["h"] is not None) else None
        shifts.append({
            "name": s["name"],
            "dx_m": dx, "dy_m": dy, "horizontal_shift_m": (dx * dx + dy * dy) ** 0.5,
            "dh_m": dh,
            "std_x_m_base": b.get("std_x_m"), "std_x_m_preview": s.get("std_x_m"),
            "std_y_m_base": b.get("std_y_m"), "std_y_m_preview": s.get("std_y_m"),
            "std_h_m_base": b.get("std_h_m"), "std_h_m_preview": s.get("std_h_m"),
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
            "degrees_of_freedom": {
                "base": bstat["degrees_of_freedom"],
                "preview": pstat["degrees_of_freedom"],
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
            "loop_closures": closure_changes,
            "station_shifts": shifts,
        },
    }
