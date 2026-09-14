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


class Baseline(BaseModel):
    """GNSS 三维基线向量（接收机原始输出，三分量相关）。"""
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = Field(None, description="基线编号，缺省自动生成 b1/b2…")
    frm: str = Field(..., alias="from", description="基线起点")
    to: str = Field(..., description="基线终点")
    dx: float = Field(..., description="基线向量 x 分量 x(to)-x(from)")
    dy: float = Field(..., description="基线向量 y 分量 y(to)-y(from)")
    dh: float = Field(..., description="基线向量高程分量 h(to)-h(from)")
    covariance: list[list[float]] = Field(
        ..., description="完整 3×3 协方差阵 [[cxx,cxy,cxh],[cyx,cyy,cyh],[chx,chy,chh]]"
    )
    unit: Optional[Literal["m", "km", "ft", "us-ft"]] = Field(
        None, description="分量与协方差的长度单位；缺省取 units.distance"
    )
    weight: float = Field(1.0, ge=0.0, description="额外权重因子（乘在先验精度上）")


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
    baselines: list[Baseline] = Field(
        default_factory=list,
        description="GNSS 三维基线（dx/dy/dh + 完整 3×3 协方差阵，相关观测）",
    )
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
    disable_baselines: list[str] = Field(
        default_factory=list, description="临时停用的 GNSS 基线 id"
    )
    baseline_covariance_scale: Optional[float] = Field(
        None, gt=0.0,
        description="所有参与预演的 GNSS 基线，其完整 3×3 协方差阵整体乘以该倍数"
                    "（>1 放宽，<1 收紧）",
    )
    baseline_weight_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="基线 id -> 新的额外权重因子（0 表示停用）",
    )
    baseline_covariance_scales: dict[str, float] = Field(
        default_factory=dict,
        description="基线 id -> 该基线协方差阵单独乘以的倍数",
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
    baselines: list[Baseline] = Field(
        default_factory=list,
        description="该期 GNSS 三维基线（dx/dy/dh + 完整 3×3 协方差阵）",
    )


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


# ---------------------------------------------------------------------------
# 测前网形设计（first-order design / reliability design）
# ---------------------------------------------------------------------------

class DesignPoint(BaseModel):
    """待定点的近似坐标（与近似高程）。"""
    name: str = Field(..., description="待定点名（唯一，且不能与已知点重名）")
    x: float = Field(..., description="近似 x 坐标")
    y: float = Field(..., description="近似 y 坐标")
    h: Optional[float] = Field(None, description="近似高程；参与高差/基线时建议给出")


class DesignObservation(BaseModel):
    """候选全站仪观测边：可包含方向角/平距/高差的任意组合。"""
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = Field(None, description="候选观测编号，缺省自动生成 o1/o2…")
    frm: str = Field(..., alias="from", description="测站")
    to: str = Field(..., description="照准目标")
    azimuth: Optional[float | str] = Field(None, description="候选方向角 from→to")
    distance: Optional[float] = Field(None, description="候选水平距离")
    dh: Optional[float] = Field(None, description="候选高差 h(to)-h(from)")
    std_azimuth: Optional[float] = Field(
        None, description="方向角先验标准差，单位取 units.angle（DMS 时为秒）"
    )
    std_distance: Optional[float] = Field(None, description="距离先验标准差（长度单位）")
    std_dh: Optional[float] = Field(None, description="高差先验标准差（高程单位）")
    cost: float = Field(1.0, ge=0.0, description="该候选观测的单次成本")
    required: bool = Field(False, description="必选观测（锁定入选，结构阶段优先纳入）")
    excluded: bool = Field(False, description="强制不选（锁定剔除）")


class DesignBaseline(BaseModel):
    """候选 GNSS 三维基线（dx/dy/dh 缺省由近似坐标反算，协方差必须完整 3×3）。"""
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = Field(None, description="候选基线编号，缺省自动生成 b1/b2…")
    frm: str = Field(..., alias="from", description="基线起点")
    to: str = Field(..., description="基线终点")
    dx: Optional[float] = Field(None, description="基线 x 分量；缺省按近似坐标反算")
    dy: Optional[float] = Field(None, description="基线 y 分量；缺省按近似坐标反算")
    dh: Optional[float] = Field(None, description="基线高程分量；缺省按近似高程反算")
    covariance: list[list[float]] = Field(
        ..., description="完整 3×3 先验协方差阵（接收机输出）"
    )
    unit: Optional[Literal["m", "km", "ft", "us-ft"]] = Field(
        None, description="分量与协方差单位；缺省取 units.distance"
    )
    weight: float = Field(1.0, gt=0.0, description="额外权重因子")
    cost: float = Field(1.0, ge=0.0, description="该候选基线的单次成本")
    required: bool = Field(False, description="必选基线（锁定入选）")
    excluded: bool = Field(False, description="强制不选（锁定剔除）")


class AccuracyGoals(BaseModel):
    """精度/可靠性目标；任一限制不给表示不约束该指标。"""
    max_horizontal_std_m: Optional[float] = Field(
        None, gt=0.0, description="待定点平面点位中误差 σp=√(σx²+σy²) 上限（米）"
    )
    max_ellipse_semi_major_m: Optional[float] = Field(
        None, gt=0.0,
        description="置信误差椭圆长半轴上限（米，置信度取 confidence）",
    )
    max_height_std_m: Optional[float] = Field(
        None, gt=0.0, description="待定点高程中误差上限（米）"
    )
    min_redundancy: Optional[float] = Field(
        None, gt=0.0, le=1.0,
        description="全部入选观测（基线按最小冗余特征值）的冗余度下限",
    )
    max_mdb_azimuth_sec: Optional[float] = Field(
        None, gt=0.0, description="方向角最小可探测粗差（MDB）上限（角秒）"
    )
    max_mdb_distance_m: Optional[float] = Field(
        None, gt=0.0, description="距离 MDB 上限（米）"
    )
    max_mdb_height_m: Optional[float] = Field(
        None, gt=0.0, description="高差 MDB 上限（米）"
    )
    max_mdb_baseline_m: Optional[float] = Field(
        None, gt=0.0,
        description="GNSS 基线三维备择检验 MDB（白化空间欧氏模）上限（米）",
    )
    points: Optional[list[str]] = Field(
        None, description="点位精度目标只约束这些点；缺省约束全部待定点"
    )


class NetworkDesignRequest(BaseModel):
    """测前网形设计请求。"""
    name: Optional[str] = Field(None, description="设计方案名称")
    known: list[KnownPoint] = Field(default_factory=list, description="已知点（固定基准）")
    points: list[DesignPoint] = Field(
        default_factory=list, description="待定点近似坐标与近似高程"
    )
    observations: list[DesignObservation] = Field(default_factory=list)
    baselines: list[DesignBaseline] = Field(default_factory=list)
    units: UnitSpec = Field(default_factory=UnitSpec)
    accuracy: AccuracySpec = Field(default_factory=AccuracySpec)
    confidence: float = Field(0.95, gt=0.0, lt=1.0, description="误差椭圆置信度")
    significance_alpha: float = Field(
        0.05, gt=0.0, lt=1.0, description="显著性水平 α（标量 Baarda 双侧、基线 χ² 单侧）"
    )
    power: float = Field(
        0.80, gt=0.0, lt=1.0, description="检验功效 1−β（决定非中心参数 λ0）"
    )
    budget: Optional[float] = Field(
        None, ge=0.0, description="总预算上限（成本单位与各候选 cost 一致）"
    )
    goals: AccuracyGoals = Field(default_factory=AccuracyGoals)
    lock_included: list[str] = Field(
        default_factory=list,
        description="额外锁定入选的候选 id（等同 required，便于复演时加锁）",
    )
    lock_excluded: list[str] = Field(
        default_factory=list,
        description="额外锁定剔除的候选 id（等同 excluded，便于复演时去锁）",
    )
    save: bool = Field(True, description="是否把设计方案与版本写入 SQLite")


class DesignReplayOptions(BaseModel):
    """按历史版本重演网形设计时允许覆盖的选项。"""
    budget: Optional[float] = Field(None, ge=0.0, description="覆盖预算上限")
    lock_included: list[str] = Field(default_factory=list, description="追加锁定入选")
    lock_excluded: list[str] = Field(default_factory=list, description="追加锁定剔除")
    unlock_required: list[str] = Field(
        default_factory=list, description="解除原请求中候选的必选标记"
    )


class DesignCompareRequest(BaseModel):
    """两套网形设计对比请求（均不落库）。"""
    design_a: NetworkDesignRequest
    design_b: NetworkDesignRequest
