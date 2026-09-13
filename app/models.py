"""请求 / 响应数据模型（Pydantic v2）。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class UnitSpec(BaseModel):
    angle: Literal["rad", "degree", "gon", "dms"] = "degree"
    """方向角的输入单位；观测行可用字符串单独给 DMS。"""
    distance: Literal["m", "km", "ft", "us-ft"] = "m"
    height: Literal["m", "km", "ft"] = "m"
    covariance: Literal["m", "km", "ft"] = "m"
    """协方差/误差椭圆的输出单位。"""


class KnownPoint(BaseModel):
    name: str = Field(..., description="点名（唯一）")
    x: float = Field(..., description="北坐标/东坐标之一，按站点 x 分量")
    y: float = Field(..., description="与 x 正交的另一平面分量")
    h: Optional[float] = Field(None, description="已知高程；不给表示高程未知")


class Observation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = Field(None, description="观测编号，缺省自动生成 o1/o2…")
    frm: str = Field(..., alias="from", description="测站（后视起点）")
    to: str = Field(..., description="照准目标（前视终点）")
    azimuth: Optional[float | str] = Field(None, description="方向角 from→to")
    distance: Optional[float] = Field(None, description="水平距离")
    dh: Optional[float] = Field(None, description="高差 h(to)-h(from)")
    std_azimuth: Optional[float] = Field(
        None, description="方向角标准差，单位取 units.angle（DMS 时为秒）"
    )
    std_distance: Optional[float] = Field(None, description="距离标准差，单位取 units.distance")
    std_dh: Optional[float] = Field(None, description="高差标准差，单位取 units.height")
    weight: float = Field(1.0, ge=0.0, description="额外权重因子（乘在默认权重上）")


class AccuracySpec(BaseModel):
    """缺省先验精度（用于未逐条给出标准差的观测）。"""
    std_azimuth_sec: float = 5.0
    """方向角中误差（角秒）。"""
    std_dist_const: float = 0.003
    """测距固定误差（米）。"""
    std_dist_ppm: float = 5.0
    """测距比例误差（1e-6 * 距离）。"""
    std_dh: float = 0.005
    """高差中误差（米）。"""


class AdjustmentRequest(BaseModel):
    name: Optional[str] = Field(None, description="方案名称")
    known: list[KnownPoint] = Field(default_factory=list)
    stations: list[str] = Field(default_factory=list, description="待求站点名")
    observations: list[Observation] = Field(default_factory=list)
    units: UnitSpec = Field(default_factory=UnitSpec)
    accuracy: AccuracySpec = Field(default_factory=AccuracySpec)
    outlier_threshold: float = Field(
        3.0, gt=0.0, description="标准化残差粗差判别阈值（常用 2.5~3.0）"
    )
    confidence: float = Field(0.95, gt=0.0, lt=1.0, description="误差椭圆置信度")
    max_iterations: int = 30
    convergence_tol: float = 1e-8
    save: bool = Field(True, description="是否把方案与版本写入 SQLite")


class PreviewRequest(BaseModel):
    """预演：在原案上临时剔除观测 / 调整权重，不落库。"""
    base: AdjustmentRequest
    disable: list[str] = Field(default_factory=list, description="临时剔除的观测 id")
    weight_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="观测 id -> 新的额外权重因子（0 表示剔除）",
    )
    std_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="观测 id -> 该观测所有分量统一乘以的标准差倍数",
    )


# ---------------------------------------------------------------------------
# 多期形变分析
# ---------------------------------------------------------------------------

class EpochObservations(BaseModel):
    """一期观测：时间、批次与该期的全部观测边。"""
    epoch: str = Field(..., description="期次标识（唯一），如 E1")
    time: str = Field(
        ..., description="观测时刻，ISO 8601（如 2026-03-01 或 2026-03-01T08:30:00）"
    )
    batch: str = Field(..., description="观测批次号")
    observations: list[Observation] = Field(default_factory=list)


class DeformationRequest(BaseModel):
    """多期形变分析请求。

    各期观测分别平差，以稳定基准点（共同控制点）为固定约束把各期坐标
    对齐到同一基准，再传播协方差并计算位移、置信椭球与显著性检验。
    """
    name: Optional[str] = Field(None, description="分析方案名称")
    datum_points: list[KnownPoint] = Field(
        default_factory=list,
        description="稳定基准点（共同控制点）及其参考坐标，至少 2 个",
    )
    monitor_points: list[str] = Field(
        default_factory=list,
        description="监测点；缺省取各期共同出现的全部非基准点",
    )
    epochs: list[EpochObservations] = Field(
        default_factory=list, description="多期观测（≥2 期，时间必须严格递增）"
    )
    epoch_selection: Optional[list[str]] = Field(
        None, description="只重算选中的两期（给 2 个期次标识）；缺省连续多期分析"
    )
    displacement_threshold: float = Field(
        0.01, gt=0.0, description="位移阈值（米），超过即认为发生形变"
    )
    confidence: float = Field(
        0.95, gt=0.0, lt=1.0, description="置信椭球与显著性检验的置信度"
    )
    units: UnitSpec = Field(default_factory=UnitSpec)
    accuracy: AccuracySpec = Field(default_factory=AccuracySpec)
    outlier_threshold: float = Field(
        3.0, gt=0.0, description="各期平差的标准化残差粗差判别阈值"
    )
    alternative_datum: Optional[list[str]] = Field(
        None, description="替代基准点（≥2 个），用于与基准方案比较结论"
    )
    save: bool = Field(True, description="是否把分析参数与版本写入 SQLite")


class DeformationRecomputeOptions(BaseModel):
    """按版本复算形变分析时允许覆盖的选项。"""
    epoch_selection: Optional[list[str]] = Field(
        None, description="覆盖期次选择（恰好两期）；缺省沿用版本保存的参数"
    )
