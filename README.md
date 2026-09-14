# 测量网平差 API（Survey Network Adjustment）

外业多站导线与控制点汇总后，一次 **前后视写反、角度单位混用或单条粗差** 就可能让闭合差
超限，却难以定位问题记录。本服务对平面导线（方位角 + 平距）、高程导线（高差）与
**GNSS 三维基线向量（dx/dy/dh + 完整 3×3 相关协方差阵）** 做统一
预检与 **加权最小二乘参数平差**，逐条给出残差、标准化残差/Mahalanobis 统计量和粗差提示，
并计算点位精度、误差椭圆与三维误差椭球；方案与计算版本写入 SQLite，可按编号复算并追溯全部参数。

- Python 3.11 · FastAPI · NumPy · SciPy · Pydantic v2
- 内部统一 SI（弧度 / 米）；支持 `degree / gon / rad / DMS`（如 `112-30-15` 或 `112°30′15″`）
  角度单位与 `m / km / ft / us-ft` 长度单位
- GNSS 基线保留接收机给出的分量相关性：平差前对每条基线做 **Cholesky 分解**，
  同时白化残差向量与雅可比（`b = L⁻¹·l`、`B = L⁻¹·J`），再与方向角/平距/高差共同组成法方程
- 内部使用测量界标准的 **Baarda w 检验**（先验标准化残差）判别粗差，避免后验 σ0 被粗差
  撑大造成的掩盖（masking）；相关基线整体用 **Mahalanobis 统计量 w² = vᵀΣ_v⁻¹v ~ χ²(3)**
  检验，同时报告后验标准化残差 t

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
| `baselines[]` | GNSS 三维基线 `from/to/dx/dy/dh` + **完整 3×3 `covariance`** 与 `unit`（缺省取 `units.distance`） |
| `units` | angle/distance/height/covariance 单位 |
| `accuracy` | 缺省先验精度：方向角中误差（秒）、测距固定误差+ppm、高差中误差 |
| `outlier_threshold` | 粗差判别阈值（标准化残差，常用 2.5–3.0） |
| `confidence` | 误差椭圆/椭球置信度（默认 0.95） |

`azimuth` 传字符串时按 DMS 解析，即使全局单位是 degree 也可以混用 `45-00-00`。

### GNSS 基线

每条基线提交起点、终点、向量三分量与接收机输出的完整协方差阵
（行/列顺序 `dx, dy, dh`，单位随分量，例如 m 与 m²）：

```json
{"id": "b1", "from": "G01", "to": "G02", "dx": 500.0, "dy": 0.0, "dh": 0.5,
 "covariance": [[2.5e-5, 3.0e-6, -1.5e-6],
                [3.0e-6, 2.5e-5,  1.0e-6],
                [-1.5e-6, 1.0e-6,  1.0e-4]]}
```

- 协方差按单位换算（分量乘 `k` 时协方差乘 `k²`），**检查对称性**（相对容差 1e-9）
  与 **正定性**（Cholesky；失败返回特征值）；基线同时进入网络连通性、初值双向传播与秩亏诊断
- 纯 GNSS 基线网 **1 个已知三维点** 即可固定全部 7 个基准亏缺（基线向量自带尺度与方向）；
  与全站仪网混合时，不含基线的平面分量仍需 2 个已知点固定旋转；基线端点若是已知点则必须给 `h`
- 响应 `baselines[]` 逐条给出：**残差向量**（米与输入单位）、残差模、
  先验/后验 **Mahalanobis 统计量** 与 χ²(3) p 值、各分量先验 Baarda w、
  冗余度与杠杆值、**协方差贡献**（信息矩阵 `w·C⁻¹` 与去掉该基线后端点协方差的缩减量）、粗差标记
- 基线上端点的 x/y/h 联合估计，`stations[]` 增加完整 **3×3 点位协方差**与 95% 三维误差椭球

## 预检（失败返回 400 与错误码）

- `unknown_endpoint` / `duplicate_point` / `self_loop` / `empty_observation` / `bad_std`
- 平面网 **≥2 个已知点**（纯全站仪网）；含 GNSS 基线的分量 **≥1 个已知三维点**
  （`insufficient_datum`）；高程网 **≥1 个已知高程点**（`insufficient_height_datum`）
- 基线已知端点必须有高程（`baseline_endpoint_without_height`）
- 基线协方差形状不是 3×3（`bad_covariance_shape`）、不对称（`covariance_not_symmetric`）、
  非正定（`covariance_not_positive_definite`，附特征值）
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
`std_overrides` 缩放标准差，返回与原案的对比：自由度、秩与缺秩、参数个数、σ0、χ²、
最大标准化残差/Baarda 值、各环平差后闭合差以及 **点位位移（dx/dy/dh）与精度变化**，
不写库、不改原案。

GNSS 基线相关参数：

- `disable_baselines: ["b1"]` 临时停用基线（自由度每条减少 3；停用后若基准不足会返回 400）
- `baseline_covariance_scale: 4.0` 对全部基线的完整 3×3 协方差阵 **整体缩放**
  （>1 放宽/降权，<1 收紧；`scale=4` 相当于所有分量标准差翻倍）
- `baseline_covariance_scales: {"b2": 9.0}` 逐条缩放；与整体缩放连乘
- `baseline_weight_overrides: {"b1": 0.0}` 基线调权（0 等同于停用）

对比结果中 `baseline_count`、`rank`、`rank_deficiency`、`parameters` 随之更新，
便于比较加入/停用基线或改变其先验精度前后的坐标、点位精度、自由度与秩。

## 多期形变分析

同一控制网在不同日期复测时，单看两期坐标差会把仪器噪声误当成位移。
`POST /api/v1/deformations` 对多期观测做完整的形变分析：

**请求**：`datum_points`（稳定基准点/共同控制点及参考坐标，≥2）、`epochs[]`
（每期 `epoch`/`time`/`batch`/`observations`，并可选提交 `baselines` GNSS 三维基线，
时间须严格递增）、
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

# 全站仪 + GNSS 三维基线联合平差
curl -s localhost:8000/api/v1/adjustments -H 'Content-Type: application/json' \
  -d @examples/gnss_baseline_request.json | python3 -m json.tool | less

curl -s localhost:8000/api/v1/deformations -H 'Content-Type: application/json' \
  -d @examples/deformation_request.json | python3 -m json.tool | less

pytest -q
```
