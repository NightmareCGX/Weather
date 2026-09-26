# Weather Platform Sharded V2 Final Validation & Implementation Plan

**性质**:Goal A(cloud_ceiling 语义验证收口)+ Goal B(可直接实施的 Sharded V2 Implementation Plan)。本轮零 production code 修改。前两轮证据见 [`REPORT.md`](./REPORT.md) 与 [`QUANTIZATION_BENCHMARK.md`](./QUANTIZATION_BENCHMARK.md);本文不重新展开 benchmark。

---

## 1. Executive Summary

- **cloud_ceiling 最终结论:Decision B —— 保持 float32(semantic compatibility exception)**。真实数据实测 4 个场(GFS f006/f120、GEFS 0.25° 成员 gep01、GEFS 0.25° geavg 均值)**全部出现真实 sentinel 翻转**(122/112/140/203 个格点,占 finite 0.011–0.020%,方向全部 unlimited→finite)。与降水相反,这里**基线是政策正确方**(真值 19.991 km ≥ 19.99 → unlimited,f32 精确保留),f16 才是偏离方;受影响格点的产品输出会从"unlimited"变成"19.984 km 的有限云底高",不可接受。存储代价 ≈ 全周期 +0.1–0.2%。
- **sharded_v2 representation 正式冻结**(§4/§5):连续量 plain `<f2` + 两个 f32 语义例外(precipitation_amount_3h、cloud_ceiling)+ det/mem flags `u1` + mean flags `<f4`;Zstd L5;index/trailer/对象布局与 v1 完全相同;版本仅由 manifest `storage_format_version: "sharded_v2"` 区分。
- **可以进入 implementation**。本计划给出 file/function 级变更矩阵(§19)、8 个 Phase(§20)、5 个 GO/NO-GO 闸门(§15)、无数据迁移的回滚合同(§16)。**无剩余 technical blocker**(§21/§22):所有前两轮识别的 blocker 要么已收口(ceiling 决策、flags 短路修复方案、writer guard 设计),要么本就是产品决策项且已给出默认路径。

## 2. Frozen Decisions from Prior Investigations

| 决策 | 依据 |
|---|---|
| continuous = plain float16(`<f2`)+ Zstd L5 | QUANTIZATION_BENCHMARK:plain f16 处于 Pareto 前沿(全周期 −22.4%);quantized 变体 ≤+3.6% 相对且带永久策略成本;f32 容器内任何量化 ≤8.3%(G11) |
| precipitation_amount_3h = f32 | exactly-0.10mm 阈值:GEFS 十进制打包产生海量恰等于 f32(0.1) 的格点(12.813%),f16 会在生产 float64 比较域翻转湿→干 |
| **cloud_ceiling = f32(本轮新增)** | 见 §3 |
| det/mem flags = u8;mean flags = f32 | flags 是 0/1 类别;mean flags 是成员均值概率(0..1) |
| 不采用 | decimal quantization、variable-specific quantization、quantized f16、mantissa grooming、fixed-point、global round/truncate |
| compute 域不变 | f16 算术 8.7× 慢 + reduction 溢出;float64 合同(deaccum/cloud/stats)保留 |
| chunk cache = native persisted dtype | 只有 native 缓存能兑现 RAM 收益(20,000 vs 40,000 B/chunk) |
| v1 字节合同完全可读;同 cycle 禁止切换格式;无 DB migration;无 bulk conversion | manifest 机制 + per-cycle store 隔离(REPORT.md §16/§20) |

## 3. Cloud Ceiling Targeted Validation(Goal A)

### 3.1 Authoritative 链路与比较点(file:function:line,实测代码)

| 层 | 位置 | 语义 |
|---|---|---|
| GRIB 源 | `parser.py:143-149`(`gh`,`typeOfLevel=cloudCeiling`) | gpm |
| 归一化 | `pipeline.py:269-273`(gpm→km,÷1000,float32 域除法) | km |
| 常量 | `cloud.py:28-29`:`THRESHOLD_KM=19.99`,`SENTINEL_KM=20.0` | |
| domain 分类 | `cloud.py:253` `classify_cloud_ceiling`:`value_km >= threshold` → unlimited | Python float(f64) |
| point serving | `point_forecast.py:588-597`:`val_km = float(raw_ceil)`;`val_km >= 19.99` → `ceiling=None, unlimited=True` | **float64 域** |
| tiles 渲染 | `tiles.py:766`:`finite = isfinite & valid & (values < 19.99)`(float64 window);unlimited → 透明像素 | float64 |
| tiles legacy 成员路径 | `tiles.py:1263`:`raw_members >= 19990.0`(**单位 bug:值是 km,永 False**);`tiles.py:1307`:wrap 分支用 `19.99`(km,正确) | float64;**两分支不一致** |
| ensemble 摘要 | `cloud.py:340-343`(`val = float(m); val >= threshold` → unlimited 计数)、`:354`(条件分位 finite 集合)、`cloud.py:383-423`(low-ceiling probability) | float64 |
| ensemble PDF | `ensemble_data.py:610`:`[m for m in members if m < 19.99]` | float64 |
| 前端 | `frontend/src/lib/forecast/labels.ts:114`:`19.99 * 3280.84`(ft) | JS Number(f64) |

**比较域判定**:所有活跃比较都在 reader 边界 `float()` 提升后的 float64 域——与 precipitation 0.10 完全同构,不存在 np.float32 域比较的歧义点;风险集中在 **f16 量化是否把值移过 f64 的 19.99**。

### 3.2 Sentinel 语义(以代码为准)

`>= 19.99 km → unlimited`(严格大于等于;`== 19.99` 也算 unlimited)。无 `> 19.99` 或 `== 20.0` 变体。f32(19.99)=19.9899997711 在 float64 域 **< 19.99 → finite**(基线对"恰好 19.99 km"的真实分类是 finite——这是 f32 展开落在阈值下方的二阶效应,实测确认)。

### 3.3 真实数据(本轮最小下载:gfs_f006/f120 + GEFS 0.25° gep01/geavg f120 ceiling;GEFS 0.5° pgrb2a 无此字段,平台实际走 pgrb2sp25 0.25°)

| field | km min..max | ≥19.99 格点 | [19.989,19.995] 带内值 | **f16 flips** | 方向 |
|---|---|---|---|---|---|
| gfs f006 | 0.009..19.9999 | 491,517 | 354 | **122(0.0118%)** | unlimited→finite |
| gfs f120 | 0.009..20.0001 | 462,858 | 325 | **112(0.0108%)** | unlimited→finite |
| GEFS gep01 (0.25°) | 0.010..19.9999 | 577,017 | 472 | **140(0.0135%)** | unlimited→finite |
| GEFS geavg (0.25°) | 0.215..20.0001 | 231,247 | 594 | **203(0.0196%)** | unlimited→finite |

GRIB ceiling 打包 ≈0.4 m 量化,在 sentinel 下方密集取值(19.900, 19.9006, …);翻转窗口 = 源值落在 **[19.9900000149, 19.9921875) km**(f16 栅格 {19.984375, 20.0} 的半格),即 **≈2.19 m 宽,且真实数据命中**(gpm 19990–19992 一带)。

### 3.4 对抗边界重放(生产 float64 域,逐值)

gpm 19985–19990 → 双方 finite;**19991/19992 → 基线 UNLIM / f16 finite(FLIP)**;19993–20001、20.0/20.01/20.02 → 双方 UNLIM(f16 上取整到 20.0 精确表示);python-float 语义下 19.95–20.02 逐点重放仅 19991/19992 带翻转。无 finite→unlimited 反向翻转(f16 上取整跨阈需要真值 ≥19.9921875 > 19.99,基线已是 unlimited)。

### 3.5 Ensemble 语义影响

`cloud_ceiling_ensemble_summary` 的 unlimited 计数、`N_finite`、条件分位成员集合、`compute_low_ceiling_probability` 的 valid 分母、PDF 过滤(`ensemble_data.py:610`)全部由同一 `>= 19.99` 谓词驱动——成员级翻转(gep01 140 格点)会直接移动受影响 cell 的 `unlimited_probability` 与条件分位成员集。

### 3.6 决策:**B — semantic exception,cloud_ceiling = float32**

理由:基线在此窗口内是政策正确方(`>= 19.99 → unlimited`,真值 19.991 满足),f16 是偏离方;受影响格点输出从"unlimited=true, ceiling=null"变为"ceiling=19.984 km(≈65,545 ft)"——产品可见。不构成 Decision C(基线无 float 噪声错误,无需 product-semantics follow-up;但 **`tiles.py:1263` 的 19990.0 单位 bug 是独立 bug**,列入 §18 文档/修复清单,v2 范围外)。

**通用规则(写入 v2 dtype 政策)**:任何其产品谓词与存储值直接比较的变量(threshold-coupled),必须满足二者之一:(a) 存储 f32;(b) 实证确认真实值分布与阈值之间有 >1 个 f16 ulp 的隔离带。precipitation(0.10)与 cloud_ceiling(19.99)均不满足 (b) → f32。

## 4. Final Variable/Dtype Matrix(v2 冻结)

| Variable / field class | v1 stored dtype | **v2 stored dtype** | product_role 差异 | Reason |
|---|---|---|---|---|
| temperature_2m | f32 | **f16** | 无 | safe continuous |
| relative_humidity_2m | f32 | **f16** | 无 | safe continuous |
| wind_u_10m / wind_v_10m | f32 | **f16** | 无 | safe continuous |
| wind_gust | f32 | **f16** | 无 | safe continuous |
| precipitation_rate | f32 | **f16** | 无(GEFS 无此变量) | safe continuous |
| cloud_cover_3h | f32 | **f16** | 无 | safe continuous(reconstruction 误差 ≤0.1pp,REPORT.md §7) |
| snow_depth | f32 | **f16** | 无 | safe continuous |
| visibility | f32 | **f16** | 无 | safe continuous |
| **precipitation_amount_3h** | f32 | **f32** | 无 | exactly-0.10mm threshold compatibility |
| **cloud_ceiling** | f32 | **f32** | 无 | 19.99km sentinel compatibility(§3) |
| crain / csnow / cfrzr / cicep | f32(实际字节) | **u8** | **det/mem** | categorical |
| crain / csnow / cfrzr / cicep | f32 | **f32** | **ensemble_mean** | 成员均值概率 0..1 |
| shard index | `<u8` ×2 | **`<u8`(不变)** | — | dtype 无关 |
| trailer | `<u8 ×2 + <u4 ×1` | **不变** | — | dtype 无关 |

## 5. Sharded V2 Storage Contract(冻结)

1. **Payload dtype**:显式 little-endian——f16=`<f2`、f32 例外=`<f4`、categorical=`u1`。禁止 platform-native 歧义端序(v1 的 `np.float32` 隐含 native;v2 修复)。
2. **Compression**:Zstd level 5,逐 100×100 chunk 独立压缩(与 v1 相同,`zarr_writer.py:628`)。
3. **Chunk geometry**:100×100;边缘 chunk 保持 v1 的 `min(100, size)` 不填充特例(`zarr_writer.py:642-645`)。
4. **Index/trailer/对象布局**:**完全保持 v1**——magic `0x53484152` 不变、16B index entry 不变、12B trailer 不变、对象键 `{var}/shard.{det|mean|mem%03d}_L%04d.shard` 不变。理由:index/trailer 数学 dtype 无关(offset/length 来自运行时 index);改 magic 只会破坏 inventory/工具兼容,无收益。
5. **版本区分**:仅 manifest `storage_format_version: "sharded_v2"`(`coordinator.py:987,1256` 写入;`manifest_reader.py:171-184` 透传,无白名单,天然支持)。注意命名风险:legacy fallback 字符串 `v2_unsharded` 与新 `sharded_v2` 相似但完全不同——legacy 字符串冻结不改,代码注释标注。
6. **Metadata 一致性**:`prepare_run_store`(`zarr_writer.py:373`)预分配 `.zarray` 时使用 resolver 解析的**真实存储 dtype**(修复 v1 的 flags 元数据 uint8 vs 字节 f32 不一致)。**不新增 per-variable dtype manifest 字段**:dtype 由冻结的 `(format_version, variable, product_role)` resolver 确定性推导,resolver 与 format 常量同版本化("resolver 即 metadata"原则);`.zarray` 作为一致性断言面(见 §19 测试)。

## 6. Dtype Resolver Design(product-role aware)

复用现有 product 身份枚举 **`TARGET_KIND_DET / TARGET_KIND_MEAN / TARGET_KIND_MEM`**(`domain/reclamation.py:19-21`,值 `"det"/"mean"/"mem"`,DB CHECK 同构 `entities.py:366-370`)——不创造新概念。

```python
# packages/domain/src/domain/storage_dtype.py(新,纯 domain,双侧可依赖)
class ProductRole(StrEnum):        # 直接对齐 TARGET_KIND_*
    DETERMINISTIC = "det"
    ENSEMBLE_MEMBER = "mem"
    ENSEMBLE_MEAN = "mean"

#: sharded_v2 冻结矩阵(与本文 §4 一一对应)
_V2_FLOAT16: frozenset[str] = frozenset({...9 个连续量...})
_V2_F32_EXCEPTIONS: frozenset[str] = frozenset({"precipitation_amount_3h", "cloud_ceiling"})
_FLAG_VARIABLES: frozenset[str] = frozenset({"crain", "csnow", "cfrzr", "cicep"})

def resolve_storage_dtype(
    format_version: str, variable: str, product_role: ProductRole
) -> np.dtype:
    """v1 → 恒 float32(冻结);v2 → 矩阵查表;未知 variable → fail-closed raise。"""
```

设计要点:v1 路径**永远返回 float32**(保护旧 cycle);未知变量 fail-closed(新增变量必须显式入表——这正是"future variable maintenance cost"的落点,单点决策);ingestion(`zarr_writer`)、serving(`zarr_v2 reader`)、inventory、self-read 共用同一函数。

## 7. Writer Safety Contract

在 `astype("<f2")` **之前**逐 chunk 校验(`zarr_writer.py:658` 处的新 guard):

```text
NaN          allowed(missing 合法语义),单独计数,不拒绝
±Inf         reject(无论来源)
|value| > 65504.0            reject(f16 max finite;防止静默 → inf)
flags(det/mem)              值域 ⊆ {0,1} 断言(u8 化后天然成立,保留为合同断言)
mean flags                   值域 ⊆ [0,1] 断言(概率语义)
```

- 违规行为:**raise** `F16RangeViolationError(IngestionError)`(新,`core/base.py`),由现有 region/cycle 失败路径处理(同 `DeaccumulationError` 模式)——**不静默钳制、不静默 NaN 化**。
- 错误消息必须含:model、cycle_time、lead、variable、product_role(det/mean/mem)、region key、违规计数、NaN 计数、min/max。
- 检查顺序:`isfinite` 掩码分离 → NaN 计数 → Inf 检查(在 NaN 之外)→ 幅度检查(在有限值上)→ astype。
- 为什么在 astype 之前:70000 → astype(f16) → inf 会静默通过且被下游 `isfinite` 当无效丢弃,数据静默损失(实测 REPORT.md §17)。

## 8. Reader Architecture

```text
ShardedV1Reader(api/core/zarr.py:213)      → 冻结:只修严重 bug,不扩展
ShardedV2Reader(api/core/zarr_v2.py,新)   → 复用/组合:_ChunkPlacement、_chunk_placements、
                                              _bilinear、index/point cache 模式、executors
get_sharded_reader_for_format(...)          → 工厂:按 manifest_storage_format 选择类
```

- v2 差异集中在一点:`read_chunk` 的 `np.frombuffer(raw, dtype=resolve_storage_dtype(...))` + `astype`/NaN-fill 使用 native dtype;index/trailer/placement 逻辑逐行复用(v1 的 `zarr.py:316-372` 数学不变)。
- 5 个 serving 分支点(`point_forecast.py:781,1386`、`tiles.py:986-987`、`ensemble_data.py:932-935,1052-1055,1203-1206`、`vector_field.py:232-233`)从 `== "sharded_v1"` 改为调用工厂/谓词;legacy `v2_unsharded` 分支原样保留。
- `get_sharded_reader`(`zarr.py:709`)LRU key 仍为 store_path(一个 store 只有一个 format,由 manifest 决定类)。

## 9. Native-Dtype Chunk Cache Design(RAM contract)

`_chunk_cache` 缓存**反解后的 native dtype 数组**,不做"读后立即整块转 f32":

```text
f16 continuous   → cache f16(20,000 B/chunk)
u8 flags         → cache u8(10,000 B/chunk)
f32 exceptions   → cache f32(40,000 B/chunk)
```

- 典型混合(v2 全变量集)per-reader 512 chunks:≈ **10.4 MiB**(v1 19.58 MiB,−47%);point/index 缓存不变(reader 总量 −10~15%,REPORT.md §10)。
- 计算 transient(f32 window、float64 tile)与缓存分层解耦——见 §10。
- 禁止事项:`read_chunk` 返回前 `.astype(np.float32)` 全块升级。

## 10. Compute Promotion Boundaries

| path | 边界 |
|---|---|
| Point interpolation | chunk 缓存 native;**只把 4 个角点 scalar** `float()` 提升到现 compute scalar 域(f64)再 `_bilinear`(现 `zarr.py:508-511,133-149` 模式原样保留);禁止为 4 个值 cast 整块 |
| Window/map | `read_window` 组装时逐 chunk copy-in 到 **float32 working window**(现 `zarr.py:612` 语义;window 是请求级 transient,不影响缓存 RAM);tiles 端现 float64 升档不变(`tiles.py:718-720`) |
| Ensemble statistics | member native chunk → 读点后 float()/f32 → 现有 domain 统计(`_validation.py:43` f64 coerce 不变);mean flags f32 路径不变 |
| Vector field | u/v 提升点同 point;`encode_vector_field_int16` 输入域不变(`wind.py:629-665`) |

## 11. Ingestion / Writer Change Map

| File | Function | Change |
|---|---|---|
| `packages/domain/src/domain/storage_dtype.py` | 新模块 | resolver + v2 矩阵(§6) |
| `services/ingestion/src/ingestion/core/base.py` | 新异常 | `F16RangeViolationError` |
| `core/config.py:236` | `STORAGE_FORMAT_VERSION` | 默认保持 `"sharded_v1"`;切换仅经 env;新增注释:cycle 启动即冻结 |
| `core/zarr_writer.py:261` | `prepare_run_store` | 预分配 dtype = resolver 结果(修 `.zarray` 一致性) |
| `core/zarr_writer.py:615` | `encode_region_sharded_v1` | 新增 v2 编码路径(或参数化):per-chunk native buffer + writer guard(§7)+ 显式 `<f2`/`<f4`/`u1` tobytes;`build_sharded_v1_container` 复用不变 |
| `core/zarr_writer.py:408,539` | `commit_region` ×2 | format 分支加 `"sharded_v2"`(走 v2 编码) |
| `core/zarr_writer.py:900,1043` | `_populate_sharded_data` / `read_slice` | **dtype-aware frombuffer**(关键:cloud_cover_3h 的 fallback 前驱读是 f16;precipitation 前驱保持 f32——resolver 按 variable 区分,不会误判) |
| `core/coordinator.py:987,1256` | manifest ×2 | 继承 setting;新增 cycle 内格式冻结断言(首次 commit 时快照,后续 commit 校验一致,不一致 fail) |
| `core/inventory.py:606-628` | `region_expected_object_keys` | format 分支加 `"sharded_v2"`(同一 14/15 shard 计数;不改 magic 校验 `:530,564`) |
| `core/pipeline.py:766-768` | `_normalize_canonical_units` | flags 短路修复(§Part 8):`target_unit == "flag"` 时即使 token 相同也强制 `np.asarray(..., dtype=np.uint8)` |
| `core/wave_runner.py:1358-1375` | predecessor 状态 | **零改动**(内存 f32 前驱;precipitation 保持 f32 后 fallback `read_slice` 亦 f32) |
| `core/markers.py` | — | 零改动(marker 与 dtype 无关) |

## 12. API / Serving Change Map

| File | Change |
|---|---|
| `api/core/zarr_v2.py` | 新 `ShardedV2Reader`(§8) |
| `api/core/manifest_reader.py:171-184` | 无白名单 → 自动透传;仅加 docstring 与 `sharded_v2` 用例 |
| 5 个分支点(§8 列表) | `== "sharded_v1"` → `is_sharded_store_format(v)` 谓词 + 工厂 |
| `api/services/point_forecast.py / tiles.py / ensemble_data.py / vector_field.py` | 数值路径零改动(reader 已输出 float()/f32 transient);`tiles.py:1263` legacy 单位 bug 单列(§18),不在 v2 范围内顺手改 |
| `api/schemas.py` | 零改动 |

## 13. Test Migration Strategy

**V1 exact contracts 冻结(不放宽)**:`test_sharded_storage.py`(f32 fixture 写入 `:51`)、`test_serving_chunk_equivalence.py`(`abs=1e-9`,`:18-27` fixtures)、`test_serving_selection_shape.py:210`(np.float64 精确)。标记 `v1_format_only`,fixture 保持 f32。

**V2 新合同**(容差来自两轮实测上界):

| 类 | 用例 |
|---|---|
| Serialization | f16 round-trip;f32 例外 round-trip;u8 round-trip;NaN 透传;±Inf reject(raise + 消息字段);65504 边界(65504 过 / 65505 拒);显式 `<f2`/`<f4`/`u1` 端序;mixed-dtype 同 store 读取 |
| Metadata | `.zarray` dtype == resolver dtype == 实际字节;manifest `sharded_v2` 写/读;v1+v2 同进程并存 |
| Precipitation | v2 中 precip shard 字节仍 f32;exactly-0.10 行为不变(生产 float64 域);生产内存前驱路径;fallback 前驱(f32);reset 边界;clamp 三段;NaN |
| Cloud ceiling | sentinel 回归:19990/19991/19992/19993/20000 gpm 注入 → 分类与基线逐格一致(f32 例外使其恒真);unlimited_probability/条件分位不受存储影响 |
| Cloud reconstruction | f16 stored predecessor 读回 → 重建结果 vs f32 基线 ≤0.1pp;±5% 守卫三段 |
| GEFS | 30 members f16 → mean/median/std/P10/P25/P50/P75/P90(容差 ≤0.03°C / ≤0.03 pp);phase support(member u8 flags + mean f32 flags);mean flags 不被 u8 化 |
| Point/Map | f16 storage → 现 compute 域插值容差(温度 ≤0.03°C 等,bench1 上界×1.2);window 组装 f32;tile PNG 渲染等价(容差内) |
| Mixed format | v1 cycle + v2 cycle 同时 serving;cross-cycle fallback(predecessor/serving selection/valid-time/lifecycle 交互) |
| Writer guard | NaN 计数;Inf raise;>65504 raise(消息字段断言);flags ∈{0,1};mean flags ∈[0,1] |

## 14. Shadow Validation Plan

- **最小实现**:`canonical = s3://{bucket}/{model}/{date}/{HH}/cycle.zarr`(不动);`shadow = s3://{bucket}/shadow-v2/{model}/{date}/{HH}/cycle.zarr`。同一解码 Dataset 走 ingestion 库路径(`pipeline.ingest_grib_file` 的 encode/commit 段)写 shadow,**不经 `catalog.reserve_run`**——无 `model_runs` 行 → canonical serving(`catalog` 查询)与 GC/reclamation 均不可见;shadow 生命周期由脚本管理(验证后删除),不进 tombstone 体系。
- 同一 source cycle 双写:canonical v1(现网)+ shadow v2。
- 比较:(a) Storage:bytes/比率/PUT 字节;(b) Numerical:逐变量 max/MAE/P95/P99/threshold flips(precip exactly-0.1 与 ceiling sentinel 必须零翻转——f32 例外使其恒真);(c) Serving:同一 query 直接实例化 v1/v2 reader 对比 point/hourly/map/ensemble/phase/ceiling 响应;(d) Runtime:writer CPU、API RSS、Range GET bytes、cold latency(只要求无 material regression)。

## 15. Rollout Gates

| Gate | 内容 | 通过标准 |
|---|---|---|
| **GO/NO-GO 1 — Dual Reader Ready** | v1 测试全绿(v1 合同冻结);v2 unit 全绿;mixed-format 全绿;ceiling 决策关闭(本文 §3);dtype metadata 合同冻结(§5-6) | 全绿 + 冻结签字 |
| **GO/NO-GO 2 — Shadow Passed** | 完整 GFS shadow cycle + 完整 GEFS shadow cycle | 数值阈值满足;零语义回归;存储节省接近 benchmark(±3pp);RSS 无 regression;writer 吞吐无 material regression |
| **GO/NO-GO 3 — GFS Production** | canonical writer 切 v2,仅 GFS det | 至少一个完整 lifecycle/serving 周期观察 |
| **GO/NO-GO 4 — GEFS Mean** | 加 GEFS mean | 同上 |
| **GO/NO-GO 5 — GEFS Members** | 加 30 members(最大 footprint / 最大 blast radius) | 同上;容量收益兑现确认 |

## 16. Rollback Contract

- canonical v2 开启后回滚 = env 切回 `sharded_v1`:**新 cycle 回到 v1;已提交 v2 cycle 由 v2 reader 继续服务;零重写、零 DB migration、零批量转换**。架构确认:reader 按 per-store manifest 分支(§8),与 writer 设置解耦 ✓。
- 同一 cycle 内格式冻结:`coordinator.py` 首次 commit 快照 setting,后续 commit 不一致即 fail(§11);manifest 30s 正向缓存只影响观察延迟不影响正确性。
- 回滚不触碰 lifecycle/reclamation(reclamation 按 physical_key 与 `store_generation` 基线,格式无关)。

## 17. Lifecycle Compatibility(只验证)

逐项确认 format-transparent:canonical selection(按 catalog 最新可服务 run,无格式感知)✓;retirement(`deletion_started_at`/`deleted_at` fence)✓;reclamation queue(`physical_key` + `store_generation` 基线,`worker.py:184-232`)✓;physical tombstone ✓;anti-resurrection(`catalog.py:215-223` + `wave_runner.py:701-710`)✓;recovery/predecessor holds(`gc/worker.py:577-601`,变量级而非 dtype 级)✓;store generation ✓。**唯一需要代码动作的硬编码**:`inventory.py:628` 的 format 分支(§11)。未发现其他 `sharded_v1` whitelist 阻断 lifecycle。

## 18. Documentation Updates(release/docs phase,本轮不改)

1. README GEFS 分辨率(0.5° → 实现 pgrb2sp25 0.25°);2. `docs/ARCHITECTURE.md:320` 缓存尺寸(2048→512 + v2 native dtype 说明);3. `gc/finalizer.py:20` "14-day" → 1 天;4. `connector.py:71-72` "geavg out of scope" 注释删除;5. v2 storage contract 新文档(§5);6. precipitation f32 例外理由;7. cloud_ceiling f32 例外理由(§3);8. categorical flags native dtype;9. **独立 bug 单**:`tiles.py:1263` legacy 成员路径 19990.0 单位错误(km 值比较 meters,unlimited 永不触发)——v2 范围外单独修;10. `pipeline.py:252` float64 注释失真;11. `v2_unsharded` vs `sharded_v2` 命名警示。

## 19. File-Level Implementation Matrix

| Phase | File | Function/Class | Change | Risk | Tests |
|---|---|---|---|---|---|
| 0 | `ingestion/core/pipeline.py` | `_normalize_canonical_units:766` | flags 短路强制 u8 | 低 | `test_pipeline.py` flags 用例 + 新短路用例 |
| 0 | `packages/domain/.../storage_dtype.py` | 新 | resolver + 冻结矩阵 | 低 | resolver 单测(全变量 × 3 role × 2 format) |
| 0 | `ingestion/core/base.py` | 新异常 | `F16RangeViolationError` | 低 | — |
| 1 | `ingestion/core/config.py:236` | setting | 注释 + cycle 冻结说明 | 低 | `test_config.py` |
| 1 | `ingestion/core/coordinator.py:987,1256` | manifest | 格式冻结断言 | 中 | 冻结违例 raise 用例 |
| 2 | `api/core/zarr_v2.py` | `ShardedV2Reader` | 新 reader | 中 | §13 全部 v2 reader 用例 |
| 2 | `api/core/zarr.py` | `get_sharded_reader` | 工厂接线 | 低 | 工厂分发用例 |
| 2 | 5 个分支点(§8) | dispatch | 谓词替换 | 中 | mixed-format 用例 |
| 2 | `api/tests/fixtures/__init__.py` | fixture writer | v2 fixture 写入(u8/f16/f32 混合) | 中 | fixtures 自检 |
| 3 | `ingestion/core/zarr_writer.py:261,615,408,539,900,1043` | 五处 | dtype-aware 写/读 + guard | **高** | §13 Writer guard + round-trip + predecessor |
| 3 | `ingestion/core/inventory.py:606-628` | `region_expected_object_keys` | v2 分支 | 中 | inventory v2 用例 |
| 4 | shadow 脚本(新,scripts/) | shadow 双写+比较 | 不入 catalog | 低 | 影子报告断言 |
| 5-7 | 部署配置 | env | `STORAGE_FORMAT_VERSION=sharded_v2` 分环境灰度 | 低 | — |

## 20. Phase-by-Phase Plan

**Phase 0 — Semantic/Hardening Prerequisites**(无字节变更):flags 短路修复;resolver 模块;异常类;dtype matrix 冻结(本文 §4 签字)。*验收*:全测试绿;resolver 覆盖率 100%。*回滚*:纯加法,直接 revert。

**Phase 1 — V2 Core Types/Constants**:`STORAGE_FORMAT_VERSION` 支持值扩展(仍默认 v1);manifest 冻结断言;`.zarray` dtype 接线(仅 v2 分支生效)。*验收*:v1 全绿(字节不变)。*回滚*:revert。

**Phase 2 — Dual Reader**:`ShardedV2Reader` + 工厂 + 5 分支点 + v2 fixtures;v1 路径冻结标记。*验收*:GO/NO-GO 1。*回滚*:reader 为纯加法。

**Phase 3 — V2 Writer**:writer dtype 路径 + guards + self-read dtype-aware + inventory 分支;**canonical 仍 v1**。*验收*:v2 writer 单测 + 自写自读 + fixture 等价。*回滚*:revert。

**Phase 4 — Shadow Validation**:完整 GFS + GEFS shadow cycle 双写与四方比较。*验收*:GO/NO-GO 2。*回滚*:删 shadow 存储。

**Phase 5/6/7 — GFS det → GEFS mean → GEFS members**:env 灰度,每步一个完整 lifecycle 观察。*验收*:GO/NO-GO 3/4/5。*回滚*:env 切回(§16)。

**Phase 8 — Natural V1 Retirement**:旧 cycle 随 lifecycle 淘汰;v1 reader 去留不在本轮决定。

## 21. Risks

- **Real blocker**:无。(ceiling/precip 语义例外已闭环;writer guard 设计完成;format 冻结机制明确。)
- **Implementation detail**:测试合同迁移量(v1 冻结 + v2 新建,工作量最大);`read_slice`/`_populate_sharded_data` dtype-aware 改造(前驱正确性关键路径);5 个 dispatch 点的一致性;fixtures 双格式。
- **Optional follow-up**:JSON 未舍入字面量展示清理(独立 PR,REPORT.md §21);`tiles.py:1263` legacy 单位 bug;dispatch 谓词统一重构;v1 reader 未来去留。

## 22. Final Go / No-Go

**具备进入 implementation 的条件。** Q1:cloud_ceiling **不可**安全 f16(4 真实场 122-203 格点 sentinel 翻转)→ f32 例外。Q2:dtype matrix = §4。Q3:**是**——precip f32 仍是兼容性最安全默认(生产内存前驱 + fallback 前驱 + exactly-0.10 + 未来弹性四重收益,代价 0.4pp)。Q4:cache 在 `read_chunk` 返回 native 数组、`astype` 只发生在 compute 边界(§9/§10)。Q5:必须修改的硬编码 = `zarr_writer.py:658/1022/1128`(dtype)、`zarr_writer.py:373`(.zarray)、`inventory.py:628`、5 个 serving `== "sharded_v1"` 分支、`config.py:236`(支持值);`4 bytes/value` 假设仅 `zarr.py:421` 一处(payload;index PNG/wind i16 的 4 字节无关)。Q6:冻结 = `ShardedV1Reader` 全部、`build_sharded_v1_container`/`parse_sharded_v1_index`、v1 精确测试、`v2_unsharded` 字符串。Q7:§13 表。Q8:**能**——dual-reader 对 v1 cycle 完全惰性,可先于 writer 任意部署(生产验证要等 Phase 4 shadow 写入)。Q9:shadow = 独立 `shadow-v2/` 前缀 + 库路径写入 + 不入 catalog(§14)。Q10:**是**——env 切回即回滚,已提交 v2 cycle 由 v2 reader 服务,零迁移。Q11:**是**——GFS(1 shard/lead/var)→ GEFS mean(同构)→ members(30×)的 blast radius 递增顺序仍最低风险。Q12:无 technical blocker;5 个 implementation detail;3 个 optional follow-up。
