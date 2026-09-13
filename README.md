# 测量网平差 API（Survey Network Adjustment）

外业多站导线与控制点汇总后，一次 **前后视写反、角度单位混用或单条粗差** 就可能让闭合差
超限，却难以定位问题记录。本服务对平面导线（方位角 + 平距）和高程导线（高差）做统一
预检与 **加权最小二乘参数平差**，逐条给出残差、标准化残差和粗差提示，并计算点位精度
与误差椭圆；方案与计算版本写入 SQLite，可按编号复算并追溯全部参数。

- Python 3.11 · FastAPI · NumPy · SciPy · Pydantic v2
- 内部统一 SI（弧度 / 米）；支持 `degree / gon / rad / DMS`（如 `112-30-15` 或 `112°30′15″`）
  角度单位与 `m / km / ft / us-ft` 长度单位
- 内部使用测量界标准的 **Baarda w 检验**（先验标准化残差）判别粗差，避免后验 σ0 被粗差
  撑大造成的掩盖（masking）；同时报告后验标准化残差 t

## 安装与运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

数据库默认位于 `data/adjustment.db`（SQLite），可用环境变量 `ADJUSTMENT_DB` 覆盖。

## 请求结构

| 字段 | 说明 |
| --- | --- |
| `known[]` | 已知点 `name/x/y/h`（h 可省略） |
| `stations[]` | 待求站点名 |
| `observations[]` | 每条边 `from/to` + 任意分量 `azimuth/distance/dh`，可逐条给 `std_*` 与 `weight` |
| `units` | angle/distance/height/covariance 单位 |
| `accuracy` | 缺省先验精度：方向角中误差（秒）、测距固定误差+ppm、高差中误差 |
| `outlier_threshold` | 粗差判别阈值（标准化残差，常用 2.5–3.0） |
| `confidence` | 误差椭圆置信度（默认 0.95） |

`azimuth` 传字符串时按 DMS 解析，即使全局单位是 degree 也可以混用 `45-00-00`。

## 预检（失败返回 400 与错误码）

- `unknown_endpoint` / `duplicate_point` / `self_loop` / `empty_observation` / `bad_std`
- 平面网 **≥2 个已知点**、高程网 **≥1 个已知高程点**（`insufficient_datum` / `insufficient_height_datum`）
- 连通性检查（`disconnected` / `height_disconnected`）、无观测连接站点（`unobserved_station`）
- 初值无法传播（`no_initial_coordinates`）
- 法方程 SVD **秩亏诊断**（`rank_deficient`）：给出缺秩维数与零空间归因参数
- 重复观测不报错：作为独立观测按权参与，并在 `warnings` 中列出

## 计算内容

- Gauss-Newton 迭代：方位角 `atan2(Δy,Δx)`、距离 `√(Δx²+Δy²)`、高差 `Δh`
- 改正后坐标/高程、先验/后验单位权中误差 σ0、χ² 检验 p 值与临界值
- 每条观测的残差（弧度/米/输入单位）、杠杆值、后验标准化残差 t、先验 Baarda w
- 疑似粗差列表 `suspects`，并对典型错误给出提示：
  - 方位角残差约 ±180° → **前后视写反**
  - 残差在 1°–20° → **角度单位混用（degree/gon/rad）** 或照准方向错误
  - 高差超限 → 提示高差反号（前后视颠倒）
- 点位协方差阵、σx/σh、95%（可调）置信误差椭圆长短半轴与长轴方位角
- 生成树基本环的 **平差前/后闭合差**（平面坐标闭合差与环线全长比、水准环线闭合差）

## 预演（what-if）

`POST /api/v1/preview` 在原案上临时 `disable` 观测、`weight_overrides` 调权或
`std_overrides` 缩放标准差，返回与原案的对比：自由度、σ0、χ²、最大标准化残差、
各环平差后闭合差以及 **点位位移（dx/dy/dh）与精度变化**，不写库、不改原案。

## 多期形变分析

同一控制网在不同日期复测时，单看两期坐标差会把仪器噪声误当成位移。
`POST /api/v1/deformations` 对多期观测做完整的形变分析：

**请求**：`datum_points`（稳定基准点/共同控制点及参考坐标，≥2）、`epochs[]`
（每期 `epoch`/`time`/`batch`/`observations`，时间须严格递增）、
`displacement_threshold`（位移阈值，米）、`confidence`、`epoch_selection`
（给 2 个期次标识则只重算这两期，缺省连续多期）、`alternative_datum`
（替代基准点，用于基准方案比较）。

**处理**：各期分别平差（基准点固定 = 对齐到稳定基准）→ 协方差传播
`Σ_d = Σ_0 + Σ_k` → 逐点计算三维位移、χ² 缩放的**置信椭球**、
**联合显著性检验** `T = dᵀΣ_d⁻¹d ~ χ²(dim)`（并给各分量 z 检验）。

**结果**：每个站点的逐期位移分量（dx/dy/dh/d2d/d3d）、**速度**（m/年）、
**联合检验 p 值**、**首个超阈值时刻**、**参与观测** id 列表；
`datum_comparison` 给出基准方案与替代基准（经 Helmert 变换对齐回主基准、
协方差随 Jacobian 传播）下各点结论的逐项比较。

**预检错误（400，逐条定位）**：

- `insufficient_datum_points` 基准点不足（<2 个共同控制点）
- `datum_point_not_observed` 基准点未在每期观测中出现
- `epoch_time_order` 期次时间倒序或相同（指出相邻两期）
- `epoch_point_mismatch` 历期间点名不一致（逐期列出缺失点）
- `epoch_selection_invalid` / `alternative_datum_invalid` / `bad_epoch_time`

**版本**：`save=true`（默认）把完整分析参数与摘要写入 SQLite，
`GET /api/v1/deformation-schemes/{id}/versions/{no}` 按编号复取；
`POST .../recompute` 复算，请求体可给 `{"epoch_selection": ["E1","E3"]}`
覆盖为两期重算，`save=true` 时另存新版本。

## 版本与复算

- `POST /api/v1/adjustments` 默认落库：同名方案递增版本号，保存**完整请求 JSON**
- `GET /api/v1/schemes/{id}/versions/{no}` 追溯该版本所用全部参数与软件版本
- `POST /api/v1/schemes/{id}/versions/{no}/recompute?save=false` 按版本复算，
  结果中回显原始请求参数；`save=true` 时另存为新版本
- `GET /api/v1/software_versions` 返回 Python/FastAPI/NumPy/SciPy/Pydantic 版本

## 快速验证

```bash
curl -s localhost:8000/api/v1/adjustments -H 'Content-Type: application/json' \
  -d @examples/request.json | python3 -m json.tool | less

curl -s localhost:8000/api/v1/deformations -H 'Content-Type: application/json' \
  -d @examples/deformation_request.json | python3 -m json.tool | less

pytest -q
```
