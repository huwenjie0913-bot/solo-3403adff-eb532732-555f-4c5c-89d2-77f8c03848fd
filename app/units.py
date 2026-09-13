"""角度、长度、高差的单位换算。

所有内部计算统一使用 SI：角度 -> 弧度(rad)，距离/高差 -> 米(m)。
角度同时支持十进制单位关键字（``degree``/``gon``）和 DMS 字符串
（``112-30-15``、``112°30′15″``）。
"""
from __future__ import annotations

import math
import re

# ---- 角度 ----
ANGLE_UNITS = ("rad", "degree", "deg", "gon", "grad", "dms")
_DEG = math.pi / 180.0
_GON = math.pi / 200.0

_DMS_RE = re.compile(
    r"""^\s*([+-]?\d+)\s*[°d\-]?\s*
        (\d{1,2})\s*['′m\-]?\s*
        (\d{1,2}(?:\.\d+)?)\s*["″s]?\s*$""",
    re.VERBOSE,
)


def angle_to_rad(value: float | str, unit: str = "degree") -> float:
    """把角度换算为弧度。

    ``value`` 可以是数值，也可以是 DMS 字符串；若 ``unit == "dms"``，
    即使 ``value`` 是字符串也按 度-分-秒 解析。
    """
    if isinstance(value, str):
        return _dms_to_rad(value)
    u = unit.lower()
    if u == "rad":
        return float(value)
    if u in ("degree", "deg"):
        return float(value) * _DEG
    if u in ("gon", "grad"):
        return float(value) * _GON
    if u == "dms":  # 数值 + dms 没有意义，按十进制度处理
        return float(value) * _DEG
    raise ValueError(f"不支持的角度单位: {unit!r}（支持 rad/degree/gon/dms）")


def _dms_to_rad(text: str) -> float:
    """解析 ``D-M-S`` 形式的角度字符串为弧度。"""
    s = text.strip().replace("°", "-").replace("′", "-").replace("″", "")
    s = s.replace("'", "-").replace('"', "").replace("d", "-").replace("s", "")
    m = _DMS_RE.match(text.strip())
    if not m:
        # 退化：允许只给 "度-分"
        parts = [p for p in re.split(r"[-\s:]+", s) if p]
        if 1 <= len(parts) <= 3:
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                raise ValueError(f"无法解析的 DMS 角度: {text!r}")
            return _dms_parts_to_rad(nums)
        raise ValueError(f"无法解析的 DMS 角度: {text!r}（示例 112-30-15.5）")
    nums = [float(m.group(i)) for i in (1, 2, 3)]
    return _dms_parts_to_rad(nums)


def _dms_parts_to_rad(parts: list[float]) -> float:
    deg = parts[0]
    minutes = parts[1] if len(parts) > 1 else 0.0
    seconds = parts[2] if len(parts) > 2 else 0.0
    sign = -1.0 if deg < 0 else 1.0
    decimal = abs(deg) + minutes / 60.0 + seconds / 3600.0
    return sign * decimal * _DEG


def rad_to_deg(rad: float) -> float:
    return rad / _DEG


def rad_to_gon(rad: float) -> float:
    return rad / _GON


def rad_to_dms(rad: float) -> dict[str, float | int]:
    """弧度 -> 度分秒（秒保留 3 位小数，处理 60 进位）。"""
    decimal = abs(rad / _DEG)
    deg = int(math.floor(decimal))
    rem = (decimal - deg) * 60.0
    minutes = int(math.floor(rem))
    seconds = round((rem - minutes) * 60.0, 3)
    if seconds >= 60.0:
        seconds -= 60.0
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        deg += 1
    sign = "-" if rad < 0 else ""
    text = f"{sign}{deg}-{minutes:02d}-{seconds:06.3f}"
    return {"deg": deg, "minute": minutes, "second": seconds, "text": text}


# ---- 长度 ----
DISTANCE_UNITS = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "km": 1000.0,
    "ft": 0.3048,
    "us-ft": 1200.0 / 3937.0,
}


def distance_to_m(value: float, unit: str = "m") -> float:
    u = unit.lower()
    if u not in DISTANCE_UNITS:
        raise ValueError(f"不支持的距离单位: {unit!r}（支持 m/km/ft/us-ft）")
    return float(value) * DISTANCE_UNITS[u]


def m_to_distance(value_m: float, unit: str = "m") -> float:
    u = unit.lower()
    if u not in DISTANCE_UNITS:
        raise ValueError(f"不支持的距离单位: {unit!r}")
    return value_m / DISTANCE_UNITS[u]


def normalize_angle(rad: float) -> float:
    """归一化到 [0, 2π)。"""
    return rad % (2.0 * math.pi)


def angle_diff(rad: float) -> float:
    """角度差归一化到 (-π, π]。"""
    two_pi = 2.0 * math.pi
    d = rad % two_pi
    if d > math.pi:
        d -= two_pi
    return d
