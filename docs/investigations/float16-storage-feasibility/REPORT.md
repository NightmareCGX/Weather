# Weather Platform Float16 Storage / Float32 Compute Feasibility Report

**调查性质**:只调查、不实施(全仓库、全链路)。所有结论基于当前仓库实现与真实数据实测(GFS 2026-09-23/18z 0.25°、GEFS 2026-09-23/00z 成员,共 85 个真实解码字段;基准脚本与原始结果在仓库外的 `%TEMP%\f16bench\`)。

**总体结论:conditionally safe(有条件安全)** —— 在保持 float32 计算不变的前提下,将连续量场持久化存储与 API 解压缓存迁移到 float16,把 categorical 字段迁移到 uint8,在实测精度上对所有产品语义零实质影响;但对象存储/网络的实际节省约为 **22%(而非 50%)**,且存在 5 个必须在实施前处理的边界条件(见 §21)。

---

## 1. Executive Summary

**结论分类:technically safe 的核心子集 + conditionally safe 的整体迁移。**

- **推荐目标架构是 Architecture B**(§24):float32 解码/归一化/派生计算,float16 连续量持久化与解压 chunk 缓存,uint8 categorical(成员/确定性 shard),float32 插值/统计/归约。**float16 计算被实测证据直接否决**(Zen 3 上 f16 算术慢 8.7 倍且 reduction 溢出,§14/§15)。
- **精度安全(真实数据实测)**:
  - 逐变量 float16 量化误差:温度 max 0.027°C、RH/云量 max 0.03%、风 max 0.014 m/s、阵风 0.061 km/h、能见度 max 7.8 m、雪深 max ~1 mm、降水 max 0.125 mm(仅 306 mm 极端值处)。
  - **所有阈值翻转率为 0**:干湿边界 0.10 mm、静风 0.5 m/s、雾 1 km、95% 阴云、60 km/h 阵风、0°C 冰冻线(mean/median)。
  - de-accumulation:生产路径(前驱在内存 float32,`wave_runner.py:1358-1375`)**零语义翻转**,误差仅为最终增量量化(max 0.0625 mm);fallback 路径(前驱读存储)在真实 lead 对上同样零翻转;对抗性 corner case(真残差恰在 −0.5 mm 钳制边界)发生率 ~1/10⁶ 格点。
  - GEFS 30 成员统计(30 个真实成员场):mean max 0.007°C、分位数 max 0.0155°C、IQR max 0.029°C;30°C 阈值 P90 翻转 3/259,920 格点(0.001%)。
- **收益(实测,非理论)**:
  - 对象存储:Zstd-5 后仅省 **~22%**(payload 减半 ≠ 压缩后减半)。GFS 每报 0.83→0.64 GB,GEFS 每报 23.7→18.4 GB;活跃保留窗口(GFS ~40 报 / GEFS ~20 报)合计 ~507→394 GB,**省 ~113 GB**。
  - Range GET 字节:point 请求 7.8→6.4 KB(−18%),GEFS 统计请求 235→191 KB(−19%);**请求数不变**。
  - API chunk 缓存:解压 payload 精确减半(19.59→9.83 MiB/reader),但 reader 总缓存中 index/point 缓存与 dtype 无关,**reader 整体只省 ~10-15%**。
  - ingestion RSS:**几乎无收益**(解码/归一化保持 float32 时,只有 shard 编码缓冲减半)。
- **兼容性**:无需 DB 迁移——manifest 已携带 `storage_format_version`(`coordinator.py:987`),serving 已按 store 分支(`manifest_reader.py:171-184` + 5 个调用点),per-cycle 新旧格式共存有现成机制。
- **不推荐做的事**:float16 计算(Architecture C)、全局 round/truncate、把 uint8 flags 转成 float16、假设压缩后省 50%。

---

## 2. Current Numerical Architecture(当前数值架构与数据流)

```text
NOAA S3/NOMADS
  │  HTTPS byte-range GET(httpx,bytes)connector.py:562-569
  ▼
[bytes] ── 写入本地暂存文件 ──▶ ProcessPoolExecutor(decode=2 并发,decode_worker.py:89,97)
  │                              仅传文件路径;返回 pickle 的 xr.Dataset
  ▼
cfgrib decode(parser.py:209-214)━━━━━━━━━━━ 输出 float32(test_parser.py:52 断言)
  │                                            无任何 astype
  ▼
raw-normalized Dataset(≈60 MB float32 / 15 vars × 721×1440)
  │  ① precipitation de-accumulation(pipeline.py:302-370)
  │     输入显式 cast float64(331-332)→ float64 相减 → 钳制/NaN 守卫 → float32 输出
  │     生产路径前驱 = 内存 np.copy(float32)(wave_runner.py:1358-1375)
  │     fallback 路径前驱 = 重读已提交存储(read_predecessor_precipitation,pipeline.py:373-420)
  │  ② cloud cover reconstruction(cloud.py:103-171)
  │     float64 计算(160-161),±5% 守卫(24),float32 还回(188-189)
  │  ③ 单位换算(pipeline.py:251-299):K−273.15、×3600、×3.6、÷1000 —— 标量 vs float32 数组,保持 float32
  │  ④ flags 归一化:np.asarray(array, dtype=np.uint8)(pipeline.py:292-298)
  ▼
normalized Dataset(float32 连续量 + uint8 flags)
  │  sharded_v1 encode(zarr_writer.py:615-672)
  │  ━━ ⚠ 唯一硬编码 dtype:chunk_buf = np.full(..., np.nan, dtype=np.float32)(zarr_writer.py:658)
  │  ━━ ⚠ uint8 flags 在此被并回 float32 缓冲(而 .zarray 元数据仍记 uint8,zarr_writer.py:373)
  │  每 100×100 chunk 独立 Zstd level 5(zarr_writer.py:628)
  ▼
Shard 容器:payload(120 chunks)+ index(120×16B <u8)+ trailer(12B,'SHAR')
  │  单次 PUT(zarr_writer.py:829),无 multipart;对象键 {var}/shard.{det|mean|mem%03d}_L%04d.shard
  ▼
Object Storage(MinIO/S3)+ PostgreSQL catalog(无 format/dtype 列)
  ══════════════════════════ serving ══════════════════════════
  ▼
ShardedV1Reader(api/core/zarr.py):tail Range GET 读 index(316-372)→ per-chunk Range GET(413)
  │  np.frombuffer(raw, dtype=np.float32).reshape(100,100).copy()(421)━━ dtype 硬编码第二处
  ▼
_chunk_cache:512 × 100×100 float32 ndarray/reader(zarr.py:259;config.py:166)
  ├─▶ point:取 4 角点值 → float() 即变 Python float(455,508-511)→ Python float 双线性(133-149)
  │     → round(x,1)(point_forecast.py:547,581)/ 部分变量不 round(603-606)→ JSON
  ├─▶ window:read_window float32 组装(612)→ tiles 立即升 float64(tiles.py:718-720 等)→ np.interp 上色 → uint8 RGBA PNG(769)
  ├─▶ ensemble:per-member chunk GET(member executor 4 workers)→ domain 统计强制 float64(_validation.py:43)→ Python float
  └─▶ vector:u/v float32 → hypot(tiles.py:1070)→ int16 scale=0.01 量化(wind.py:664-665)→ Redis 原始字节
       flags(GEFS mean shard)= 成员均值概率 float32(0.65 等)→ round(x,4) 输出(point_forecast.py:575)
```

**dtype 转变点汇总**(每节点:输入→输出、显式/继承、是否 copy、是否压缩):

| # | 节点 | 位置 | 输入→输出 dtype | 显式? | copy/分配 | 压缩/序列化 |
|---|---|---|---|---|---|---|
| 1 | 下载 | connector.py:619,668 | bytes | — | 磁盘暂存 | 否 |
| 2 | cfgrib decode | parser.py:209-214 | GRIB→float32 | 继承(cfgrib) | Dataset 分配 | 否 |
| 3 | 跨进程返回 | wave_runner.py:1727-1729 | float32(pickle) | 继承 | 全量 pickle 拷贝 | pickle |
| 4 | de-accum | pipeline.py:331-339 | f32→**f64**→f32 | 显式 | 2 数组转换+diff+mask | 否 |
| 5 | cloud 重建 | cloud.py:160-189 | f32→**f64**→f32 | 显式 | 同上 | 否 |
| 6 | 单位换算 | pipeline.py:255-298 | f32→f32 | 继承(weak scalar) | 原地赋值 | 否 |
| 7 | flags 归一化 | pipeline.py:292-298 | f32→**uint8** | 显式 | 新数组 | 否 |
| 8 | shard 编码 | zarr_writer.py:658-660 | 一切→**float32** | 显式(硬编码) | chunk_buf | **Zstd-5/chunk** |
| 9 | shard 读 | zarr.py:413,421 | f32 bytes→ndarray | 显式(硬编码) | .copy() | Zstd 解压 |
| 10 | chunk 缓存 | zarr.py:259 | f32 ndarray | — | LRU 持有 | — |
| 11 | point 插值 | zarr.py:455,508-511,133-149 | f32→**Python float(f64)** | 显式 float() | 标量 | JSON |
| 12 | window→tile | tiles.py:718-720,769 | f32→**f64**→uint8 RGBA | 显式 | window 拷贝 | PNG(zlib-1) |
| 13 | domain 统计 | _validation.py:43 | 任意→**float64** | 显式 | 数组转换 | JSON |
| 14 | vector 量化 | wind.py:629-665 | f32→f32(hypot)→**<i2** | 显式 | 拷贝 | Redis bytes+gzip-6 |
| 15 | Redis 响应缓存 | cache.py:199 | Python float→JSON 文本 | — | — | JSON 文本 |

---

## 3. Current Storage Format(writer/reader/schema)

- **布局**(writer `zarr_writer.py:575-601`,reader `zarr.py:316-372`):payload = 120 个 100×100 chunk 的 Zstd-5 压缩字节顺序拼接;index 表 120×16B(`struct.pack("<QQ")`,little-endian uint64 offset/length);trailer 12B(num_chunks, index_size, magic 0x53484152)。index 从尾部一次 Range GET 读取(`zarr.py:342`),API reader **不校验 magic**(仅 ingestion inventory 校验,`inventory.py:530`)。
- **chunk 几何硬编码**:100×100、8×15=120 chunks(shard 覆盖 721×1440);reader `zarr.py:385-389`、writer `zarr_writer.py:642-645`、inventory `inventory.py:519` 三处独立声明。
- **dtype 元数据:不存在**。manifest(`__commit__/v1/manifest.json`)有 `storage_format_version` 但**无 dtype 字段**;`.zarray` 的 dtype 来自预分配数据集(`zarr_writer.py:373`)——**与 shard 实际字节不一致**(flags 元数据 uint8、字节 float32)。
- **dtype 硬编码位置**(v2 必改点):writer `zarr_writer.py:658`;reader `zarr.py:421`(唯一 payload 解码点)+ NaN fill `zarr.py:387,402,408,612,666`;ingestion 自检读回 `zarr_writer.py:1022,1128,1146`。
- **能否新旧 dtype 共存**:可以。分派钩子已存在——`manifest_storage_format()`(`manifest_reader.py:171-184`)在 5 个 serving 调用点分支(point_forecast.py:781,1386;tiles.py:986-987;ensemble_data.py:932-935 等),ingestion 按 `STORAGE_FORMAT_VERSION`(`config.py:236`)路由写路径(`zarr_writer.py:470-471`)。引入新版本串(如 `sharded_v2`)+ per-format dtype 表即可;index/trailer 数学与 dtype 无关(offset/length 来自运行时 index,`zarr.py:406`)。

**三个 migration option 评估**:
- **Option A**(保留 v1 加 dtype 元数据):manifest 虽是 store 级 JSON,但 v1 读取路径的 `np.frombuffer(dtype=np.float32)` 是无元数据硬编码——保留 v1 字符串却改变其字节语义会让"manifest 说 sharded_v1 就按 float32 读"的旧 reader 读错新 store。**否决**。
- **Option B(推荐)**:新版本串 `sharded_v2`(payload 为 per-variable native dtype:连续量 f16、flags u8、mean-shard flags 保持 f32 概率),写入 manifest 的 `storage_format_version`;reader 对 v2 按 dtype 表解码并立即统一为内部 float32(或 f16 缓存)。旧 cycle 的 manifest 仍是 `sharded_v1`,走旧路径,**零改动可读**。
- **Option C**:per-chunk 自描述(每 chunk 前置 dtype header)——破坏现有"index 直接给 offset"的紧凑布局且浪费每 chunk 字节。**否决**。

---

## 4. Current Variable Inventory(真实变量清单)

权威定义:`DEFAULT_VARIABLES`(`wave_runner.py:79-170`,15 个)、`SURFACE_FIELD_FILTERS`(`parser.py:44-150`)、单位变换 `pipeline.py:251-299`。GEFS 成员数 30(`coverage.py:30-33`),视界 0–240h/3h = 81 leads(`horizon.py:14-16`)。

| 变量 | GRIB | 源单位→内部单位 | 范围(实测) | ingestion 派生 | lead-0 | ensemble | 类型 | 存储 dtype 现状 |
|---|---|---|---|---|---|---|---|---|
| temperature_2m | `2t` | K→°C(−273.15) | −68.2..39.2 °C | 否 | instant | 是 | 连续 | f32 |
| relative_humidity_2m | `2r` | %→% | 5.7..100.0 %(无裁剪) | 否 | instant | 是 | 连续 | f32 |
| precipitation_amount_3h | `tp` | kg m⁻²→mm | 0..306.4 mm | **是**(deaccum) | **NaN** | 是 | 连续 | f32 |
| precipitation_rate | `prate` | kg m⁻² s⁻¹→mm/h(×3600) | 0..76.7 mm/h | 否(仅换算) | instant | **GEFS 不支持**(`idx_parser.py:726-732`) | 连续 | f32 |
| wind_u_10m / wind_v_10m | `10u`/`10v` | m/s→m/s | ±61.5 m/s | 否 | instant | 是(vector) | 连续 | f32 |
| wind_gust | `gust` | m/s→km/h(×3.6) | 0..230.8 km/h | 否 | instant | 是 | 连续 | f32 |
| visibility | `vis` | m→km(÷1000) | 0.02..24.1 km | 否 | instant | 是 | 连续 | f32 |
| snow_depth | `sde` | m→m | 0..2.38 m | 否 | instant | 是 | 连续 | f32 |
| cloud_cover_3h | `tcc` | %→%(clip 0-100) | 0..100 % | **是**(reconstruction) | **NaN** | 是 | 连续 | f32 |
| cloud_ceiling | `gh`@cloudCeiling | gpm→km(÷1000) | 0..20 km(sentinel 19.99=unlimited) | 否 | instant | 是(unlimited 概率+条件分位) | 连续+离散哨兵 | f32 |
| crain/csnow/cfrzr/cicep | 同名 | Code 4.222→**flag(uint8)** | 0/1 | 否(lead-0 置零) | **全 0 uint8** | 是(相态伙伴) | **categorical** | f32(⚠见 §18) |
| wind_speed | — | — | — | — | — | — | **serving 侧派生**(`point_forecast.py:844` hypot;`ensemble_data.py:301`) | 不存储 |

ingestion 仅有两个派生变量(precip、cloud),无 wind speed/dew point/apparent temp 计算(已 grep 证实)。**不要假设 precipitation 是唯一 special variable——cloud_cover_3h 同为 reset/reconstruction 变量**(§7)。

---

## 5. Float16 Numerical Error Analysis(逐变量,真实数据实测)

IEEE-754 binary16 半 ulp 解析表(实测验证):

| 值域 | spacing | 半 ulp(最大量化误差) |
|---|---|---|
| [0.001, 0.002) | 9.5e-7 | 4.8e-7 |
| [0.0625, 0.125)(含 0.10 阈值) | 6.1e-5 | 3.05e-5 |
| [1, 2) | 9.77e-4 | 4.9e-4 |
| [16, 32)(典型风速/降水) | 0.0156 | 0.0078 |
| [64, 128)(大累积) | 0.0625 | 0.03125 |
| [128, 256) | 0.125 | 0.0625 |
| [256, 512)(306mm 极端) | 0.25 | 0.125 |

真实字段(Canonical 单位,float32→float16→float32 往返):

| 变量(单位) | 实测范围 | max abs | MAE | P99 | 1dp 显示翻转 | 阈值翻转 |
|---|---|---|---|---|---|---|
| temperature(GFS)°C | −68.2..39.2 | 0.0273 | 0.0031 | 0.0148 | 24.8% | — |
| temperature(GEFS)°C | −60.3..40.2 | 0.0133 | 0.0030 | 0.0133 | 7.1% | — |
| RH(两模型)% | 5.7..100.0 | 0.0301 | 0.0148 | 0.030 | 0% | 95%/50% 均 0 |
| wind u/v(GFS)m/s | ±61.5 | 0.0145 | 0.0008 | 0.0036 | 0% | 静风 0.5m/s:0 |
| wind u/v(GEFS)m/s | ±27.7 | 0.0078 | 0.0007 | 0.0037 | ≤2.2% | 0 |
| gust km/h | 0.03..230.8 | 0.0612 | 0.0052 | 0.0263 | 7.0% | 60km/h:0 |
| precipitation(GFS)mm | 0..306.4 | **0.125** | 1.2e-6 | **0** | 0.001% | 0.10mm:0 |
| precipitation(GEFS)mm | 0..136.3 | 0.050 | 1.0e-4 | 0.0016 | 0% | 0 |
| visibility km | 0.02..24.1 | 0.0078 | 0.0044 | 0.0066 | 0.40% | 1km 雾:0 |
| snow_depth m | 0..2.38 | 0.00097 | 3.0e-5 | 0.00044 | 0.03% | — |
| cloud cover(GFS)% | 0..100 | 0.0250 | 0.0040 | 0.025 | 0% | 0 |

对比基准:
- **上游 GRIB 打包精度**:GFS APCP 打包量化实测 **0.125 mm**(所有值是 0.125 的整数倍,如 306.375)。float16 在 ≤256mm 的半 ulp ≤0.0625mm,**低于上游一个打包量子**;仅在 [256,512) 区间两者相等(0.125)。
- **显示精度**:API/前端按 1 位小数(°C、km/h、%)或 2 位(NumberFormat)渲染;float16 误差比显示粒度小 3-20 倍。
- 唯一"可见"效应是 1dp 显示值的舍入边界翻转(温度最高 24.8% 格点差 0.1°C)——与现有 0.1 显示舍入同量级,不改变语义;不涉及 `round()` 合同的变量(如 JSON 未舍入字段,见 §19/§21)。

---

## 6. Precipitation De-accumulation Analysis

**实现**(`pipeline.py:302-370`):`curr−pred` 显式 float64(331-332)→ 非负保留、`[−0.50,0)` 钳 0(`DEACCUMULATION_CLAMP_BOUND_MM=0.50`,`base.py:91-92`)、`<−0.50` 置 NaN → float32 输出(339)。前驱:lead%6==0 时取 `lead−3`。

**前驱来源(关键事实)**:生产 wave 路径使用**内存副本** `raw_precip_for_future = np.copy(ds["tp"].values)`(`wave_runner.py:1358-1375`,float32,不经存储);`read_predecessor_precipitation`(`pipeline.py:373-420`)只是 fallback/库路径。**因此 float16 存储迁移根本不改变生产路径的减法输入**——改变的只有:① 最终增量的存储量化;② fallback 路径读回 f16 前驱。

**Reset 语义**:APCP 每 6h 重置,减法双方是 3h/6h **窗口**累积(非自起报总量),量级上界 = 极端 6h 雨量。实测 GFS 单报最大 306.4mm(飓风)、GEFS 136.3mm。

**实验设计**(镜像实现,含真实 lead 对 + 对抗构造):

| 场景 | 架构 | max err vs 现状 | MAE | 语义翻转 |
|---|---|---|---|---|
| 真实 f189→f192(130→237mm) | B1: f16 前驱存储→f32 减→f16 存 | 0.0625 mm | 1.8e-7 | **0**(P99=0) |
| 真实 f003→f006(180→306mm) | B1 | 0.0625 mm | 3.6e-7 | 0 |
| 同上两对 | B2: f16 存储→f64 减→f16 存 | 0.0625 | 3.6e-7 | 0 |
| 真实两对 | C: f16 直接减(反面参照) | 0.1875 | 1.4e-6 | 0(仅 14 个 1dp 显示翻转) |
| **生产路径**:前驱内存 f32 → f64 减 → f32 结果 → **仅最终增量存 f16** | D | 0.0625 mm | 1.8e-7 | **0**(1M 格点,dry/wet、0→pos、NaN 全零) |
| 对抗:curr=pred+0.125mm,pred∈[32,131)mm | B1 | 0.0625 | 8.0e-4 | dry/wet 1 格 / ~46 万湿格(~2e-6):f16 前驱在 [64,128) 半 ulp=0.03125 侵蚀 0.125−0.10 余量 |
| 对抗:curr=pred(真增量 0) | B1/B2 | 0.0625 | 4.0e-4 | 0→pos 1 格(存 0.031mm,低于 trace 阈值,产品仍 dry) |
| 对抗:残差恰 −0.5mm(钳制边界) | B1/B2 | — | — | **valid→NaN 1 格 / 1e6**:`pred_stored=pred+0.03125` 使 −0.5−δ<−0.5 |
| 对抗:600.125−600.0 / 1100.125−1100.0 | B1/B2 | 0 | 0 | 0 |
| 同上 | C(反面) | 0.125 | 0.125 | **100% 增量丢失**(600.125 无法被 f16 表示→差值 0) |

**结论**:
1. **float16 存储 + float32/float64 减法是安全的**。生产路径(内存前驱)零翻转;fallback 路径在真实数据上零翻转,在 10⁶ 级对抗采样中出现 2 类 ~1e-6 率的边界事件(dry/wet 边界翻转、钳制边界 NaN)。
2. 大数相减的主导误差项是 **f16 前驱量化**(相对误差 2⁻¹¹),不是减法本身;float64→float32 的计算精度贡献(<1e-5 mm)可忽略——B1 与 B2 结果相同。
3. **不要用 float16 做减法**(C 架构在 >512mm 时整段丢失小增量)。
4. 钳制边界事件(真残差 = −0.50 恰好 + 前驱向上舍入)可把"钳为 0"变成"NaN 失效",率 ~1e-6 且仅影响单格;若要彻底消除,可在 v2 中把钳制界放宽一个前驱量化步长(如 −0.5−ulp_f16(pred)),这是实现期决策,不影响产品语义。

---

## 7. Other Derived / Reset Variables(cloud 与其他 special variables)

**cloud_cover_3h reconstruction**(`cloud.py:103-171` + `pipeline.py:588-688`):公式 `C_3h = (R·C_R − (R−W)·C(L−W))/W`,R=6,W=3 时 `2·C6 − C3`;输入显式 float64(`cloud.py:160-161`),守卫 ±5 个百分点(`CLOUD_COVER_RECONSTRUCTION_TOLERANCE_PERCENT=5.0`,`cloud.py:24`):[0,100] 保留、[−5,0)→0、(100,105]→100、越界 NaN;float32 还回(188-189)。前驱 = 存储读回(`read_predecessor_cloud_cover`,f32)或内存副本(`wave_runner.py:1364-1368`)。

**float16 传播分析**:误差 = |2·δ(curr)| + |δ(pred)| ≤ 2×0.031 + 0.031 ≈ **0.094 pp 最坏**(f16 在 [64,128) 的半 ulp),而守卫带是 ±5pp、显示精度 0.1%、ensemble min_valid 21——**4 个数量级的安全裕度**。lead%6==3 直接分支的 clip(0,100) 对 f16 无影响(边界值 0/100 精确表示)。**结论:cloud reconstruction 完全安全**,且前驱误差只放大一次(无递归——前驱自身是 3h 直接值)。

**其他 reset/derived/threshold-sensitive 变量逐项**:
- precipitation_amount_3h lead-0 NaN(f32 NaN → f16 NaN 保真,§17);flags lead-0 uint8 零——不变。
- cloud_ceiling:哨兵 ≥19.99 km(`cloud.py:28`);f16 在 [16,32) 半 ulp=0.0078 km,19.99±0.0078 边界翻转需要真实值落在 7.8m 窄带内;实测值域 0-24km,无溢出。**建议**:v2 保持 19.99 哨兵语义不变(f16 可精确表示 19.984375/20.0,注意 19.99 本身不是 f16 精确值——比较在 float64 域进行(`point_forecast.py:589`),读回后与现状一致地比较,无风险)。
- wind_speed:serving 派生(hypot);u/v 各自 f16 误差 ≤0.014 m/s → speed 误差 ≤~0.02 m/s(凸组合),§9 实测 mean-speed max 0.0097 km/h。
- 相态分类(`precipitation.py:143-314`):输入 amount + uint8 flags(≥0.5 判定,`point_forecast.py:899`)+ t2m;f16 下唯一敏感点是 amount 对 TRACE 0.10mm 的比较——§6 实测翻转 ~2e-6(仅对抗构造),真实 lead 对为零。
- 无其他 accumulated/reset 变量(grep 证实)。

---

## 8. GEFS Ensemble Statistics Analysis

**架构事实**:mean 是**上游官方 geavg 产品**(非平台计算,`connector.py:181-185`;member-wise nanmean 只是无调用者的 fallback,`zarr.py:651-689`)。domain 统计**强制 float64**(`_validation.py:43` `np.asarray(members, dtype=np.float64)`),std ddof=0、percentile method="linear"(`statistics.py:75,109`,测试固化 `test_ensembles.py:38-48`)。serving 每请求读 30 成员(member executor 4 workers,`zarr.py:567-581`),每成员每变量 ~1 chunk GET。

**实验**(30 个真实成员 t2m 场,f120;成员存储 f32 vs f16;计算 f64 与 f32 两路):

| 统计量 | f16 存储→f64 计算:max / MAE / P99(°C) | 备注 |
|---|---|---|
| mean | 0.00722 / 0.00053 / 0.00259 | |
| median | 0.01529 / 0.00200 / 0.0100 | |
| std(spread) | 0.00719 / 0.00052 / 0.00252 | |
| P10/P25/P75/P90 | ≤0.01548 / ~0.0027 / ≤0.0123 | |
| IQR | 0.02933 / 0.00318 / 0.0155 | |
| **f16→f32 计算**(tile fallback 路) | 与 f64 计算差异 max 1.8e-5 | **计算精度贡献可忽略** |

- **分类翻转**:0°C 冻结线(mean/median)**0 翻转**;P90≥30°C 3/259,920(0.001%);P10≤0°C 0;spread 1dp 显示翻转 0.5%(差 0.1°C 一档)。
- **成员排序**:f16 量化使 100% 格点的排序序列出现并列/换位——但分位数在量化样本上自洽,误差上界=量化半 ulp(实测 ≤0.0155°C)。**无 percentile 方法偏移**。
- **phase support / transition**:flags 为 uint8/整数判定(≥0.5),**完全不受**存储精度影响;amount 参与的 joint phase(`compute_joint_amount_phase_support`)经 §6 的阈值分析覆盖(0.10mm 翻转 ~0)。
- **PDF**:Gaussian KDE float64、100 点网格(`pdf.py:41-119`);成员值扰动 ≤0.015°C 使密度曲线变化远小于渲染分辨率。无独立 IQR 产品(P25/P75 仅进入带宽)。
- **结论**:float16 成员存储 + float32/64 计算下,ensemble 产品 **materially equivalent**(所有偏差 ≪ 集合离散度和显示精度)。

---

## 9. Point and Map Serving Analysis

**Point 插值**(`zarr.py:457-532`):2×2 邻域;chunk 内 1 次 chunk-GET(+1 次 tail index GET),跨 chunk 边界最多 4 次;角点值**立即转 Python float**(`float(arr[...])`,`zarr.py:508-511`),双线性核纯 Python float(`zarr.py:133-149`)。**无需整 chunk 升 dtype——读路径天然只把 4 个标量升为 float64**;f16 chunk 的角点误差=半 ulp,双线性权重和为 1,凸组合不放大。

实测(真实场,200k 随机点):

| 场 | 插值误差 max / MAE / P99 | 1dp 显示翻转 |
|---|---|---|
| temperature(°C) | 0.0253 / 0.0021 / 0.0102 | 3.3% |
| precip(mm) | 0.0647 / 8.7e-7 / 0 | 0 |
| wind-u(m/s) | 0.0074 / 0.0005 / 0.0030 | — |

point cache 缓存的正是 4 个 Python float(与存储 dtype 无关)。

**Map/Window**(`tiles.py:986-1124`):`read_window` 分配 float32 window(`zarr.py:612`),per-chunk GET 并行(2 workers);tiles 边界立即升 float64(`tiles.py:718-720,1087,1101`)→ float64 `np.interp` 上色(773-783)→ uint8 RGBA。**确定性 map 全程可保持 float16 resident 直到 tiles 的 float64 升档**(色带映射需要连续域,但 0.03% 的输入误差在色带 stop 分辨率下不可见)。风 tile:float32 hypot×3.6 → float64(`tiles.py:1068-1070`)。ensemble map:官方 mean shard 直接读(f16 后同样 ≤0.03% 色带偏移);member-stack fallback `read_ensemble_mean_window` 用 float32 nanmean(§8 证明 f32 vs f64 计算差异 1.8e-5)。**结论:deterministic window 可以 f16 resident;ensemble 计算边界必须升 float32/64——正是 Architecture B 的划分。**

---

## 10. API Memory Analysis(字节级)

配置:`API_READER_MAX_CACHED_CHUNKS=512`、`API_READER_MAX_CACHED_INDICES=16384`、`API_READER_MAX_CACHED_POINTS=32768`(均 **per-reader**,`zarr.py:241-245`);`MAX_READERS=8`(`zarr.py:704`);uvicorn 4 workers(compose)。

| 缓存 | 每 entry 内容 | 现状/reader | 候选(f16)/reader | 变化 |
|---|---|---|---|---|
| `_chunk_cache`(512) | 100×100 ndarray:40,000B payload + ~96B 对象头 | **19.58 MiB** | **9.83 MiB**(20,096B) | **−9.77 MiB(−50%)** |
| `_index_cache`(16384) | (120,2) uint64 = 1,920B + 头 | ~31.3 MiB | 不变 | 0 |
| `_point_cache`(32768) | 9 元组 key + 4 Python float(~400-800B) | ~13-26 MiB | 不变(Python float) | 0 |
| **reader 合计(估)** | | **~64-77 MiB** | ~54-67 MiB | **−10~15%** |
| **每 worker(×8 readers)** | | **~0.5-0.6 GiB** | ~0.44-0.53 GiB | **−78~95 MiB** |

- 现有注释"`~40 KB each`、~20 MB/reader"(`config.py:157-166`)仍准确;但 `~0.8 KB/entry` 的 point cache 估计(`config.py:188-207`)与 "~2 KB" 的 index 估计表明 **chunk payload 只占 reader 缓存的 ~25-30%**——"API 缓存内存减半"的说法不成立,准确说法是 **chunk-payload 减半、reader 总缓存 −10~15%、worker 总缓存 −78~95 MiB**。
- docs/ARCHITECTURE.md:320 的"2048 chunks"是过时值(实现 512)。
- window/tile 的 float64 组装是请求生命周期临时量,与 dtype 迁移无关(候选下输入减半,临时 float64 window 不变)。

---

## 11. Ingestion Memory Analysis

并发结构(`config.py:133-172`、`wave_runner.py:371`):download 8 / **decode 2** / write 4,staging 上界 = 14 items;wave ≤8 leads;GEFS wave = (1 mean + 30 members)×leads,lead-major。跨进程交接是**全量 pickle Dataset**(~60 MB float32/文件,`wave_runner.py:1727-1729`)。前驱副本 tp+tcc ≈ 2×8.3MB。

**候选架构(解码/归一化保持 float32)下**:驻留大头(解码 Dataset、pickle、前驱副本)**全部不变**;变化仅 shard 编码缓冲(120×100×100:4.61MB→2.30MB/shard,f16)与 PUT 字节。**ingestion RSS 净节省 <2-3%,不值得作为迁移动机**;也不要为省内存提前把 decode/normalize float16 化(会破坏 §6 的 float64 减法合同与 §8 的 float64 统计输入)。*不要为省内存过早 f16 化*的约束在此被量化证实。

---

## 12. Compression Benchmark(真实数据,Zstd level 5 = writer 同款 numcodecs 设置)

**Chunk 粒度(真实 Range GET 单元,100×100)**:

| 字段 | f32 压缩/chunk | f16 压缩/chunk | 节省 | f32 比率 | f16 比率 |
|---|---|---|---|---|---|
| temperature | 7.8 KB | 6.4 KB | **18.7%** | 5.1× | 3.1× |
| RH | 11.2 | 9.7 | 13.1% | 3.6× | 2.1× |
| wind u / v | 9.6 / 9.6 | 7.7 / 7.6 | 19.9% / 20.5% | 4.3× | 2.6× |
| gust | 9.0 | 8.0 | 11.2% | 4.6× | 2.5× |
| prate | 6.6 | 5.2 | 21.9% | 6.1× | 3.8× |
| precip(3h, GFS/GEFS) | 4.4 / 4.2 | 3.5 / 3.2 | 18.7% / **23.6%** | 9.5× | 5.4× |
| cloud(GFS/GEFS) | 8.6 / 6.9 | 6.8 / 5.7 | 21.1% / 17.6% | 4.7× | 3.1× |
| visibility | 8.0 | 4.4 | **45.4%** | 5.0× | 4.5× |
| snow_depth | 5.1 | 2.7 | **46.7%** | 8.1× | 7.7× |

**flags(当前被硬编码存 f32,`zarr_writer.py:658`)**:crain f32 100.3KB/shard-chunkset → **uint8 68.5KB(−31.7%)**;csnow −26.6%;cfrzr/cicep 近零(-9.0%/-5.2%);四 flag 合计 **−29.6%**。f16 化 flags 仅再省 5-11% 且语义错误——**必须 uint8**。
(整场粒度数字同趋势:temperature 0.86→0.71MB 等;encode/decode 时间见 §14。)

**关键反直觉结论:payload 减半 ≠ 压缩后减半**。f32 本身压缩比高(4-9×),压缩后尺寸由尾数噪声熵主导;实测连续量节省 **11-47%(多数 13-24%)**,不能按 50% 做容量规划。

---

## 13. Object Store and Range GET Savings

几何:GFS 15 vars × 81 leads × 1 det = 1,215 shards/cycle;GEFS 14 vars × 81 × (30 mem + 1 mean) = 35,154 shards/cycle;每 shard 120 chunks。实测 per-chunk 均值(bench4)加权:

| 对象 | f32 现状 | f16+u8 候选 | 节省 |
|---|---|---|---|
| GFS / cycle | 0.83 GB | 0.64 GB | −22.2% |
| GEFS / cycle | 23.70 GB | 18.43 GB | −22.2% |
| 活跃窗口(GFS ~40 报 + GEFS ~20 报,由 0-240h 视界 + canonical 最新选择推得,`planner.py:1-20`) | **~507 GB** | **~394 GB** | **−113 GB(−22.2%)** |

**Range GET 传输**(请求数不变,字节下降):
- point(GFS):1 chunk(7.8→6.4KB)+ index tail(1.92KB,不变)→ −18%;
- point(GEFS 读 mean shard):同上;
- ensemble 统计:30 chunk GET 235→191KB(−19%);precip phase:180 chunk GET(~30 成员×6 场)≈ 1.2→0.97MB;
- map tile:z≥4 一 chunk,低 zoom 至 120 chunks;
- **GEFS cold-serving 的债务本质是 request-count 与冷索引 RTT,不是字节**——f16 降低每请求字节 ~19% 和解压 CPU(payload 减半),但 **Range GET 次数完全不变**;不能宣称延迟 −50%。冷请求的 decode 时间(§14 frombuffer f16 反而快 2×)与传输字节的收益对 cold latency 是二阶改善。

---

## 14. CPU / x86 / ARM64 Analysis

实测平台:AMD Ryzen 7 5800X(Zen 3,x86-64,AVX2,**无原生 FP16 算术**;ARM64 NEON 有原生 f16 向量但 NumPy 当前同样非向量化)。numpy 2.5.3:

| 操作(4M 元素) | f32 | f16 | f16 存储→f32 计算 | 结论 |
|---|---|---|---|---|
| 减法 | 2.94 ms | **25.6 ms(8.7×慢)** | 3.48 ms | f16 计算不可接受 |
| 乘法 | 2.55 | 25.0(9.8×) | — | 同上 |
| mean | 2.38 | 6.79(2.9×)⚠**溢出** | 2.12 | f16 reduction 数值危险 |
| std | 8.92 | **61.8(6.9×)** | — | 同上 |
| percentile | 31.3 | 38.1(1.2×) | — | 排序主导 |
| astype f32→f16 | — | 10.9 ms | — | writer 量化成本 ~370M elem/s |
| astype f16→f32 | — | 8.0 ms | — | reader 升档成本 |
| frombuffer+copy | 2.34 ms | **1.18 ms(2×快)** | — | **f16 chunk 解码更快**(字节减半) |
| hypot | 12.8 | — | — | 保持 f32 |

- **f16 reduction 溢出实测**:4M 元素 N(10,20) 序列 `a16.mean()` 触发 `overflow encountered in reduce`(f16 累加器上溢 inf)——f16 计算不仅慢而且**数值破坏**。
- **目标架构应明确定位 float16 为 storage/cache/bandwidth 优化格式,不是 compute format**——本基准是直接证据。ARM64 上 NumPy 2.x 的 f16 路径同样非向量化(升级到有 f16 向量化栈前,该结论跨架构成立)。

---

## 15. Float64 Audit

| float64 使用点 | 位置 | 分类 | 处置 |
|---|---|---|---|
| de-accumulation 减法 | `pipeline.py:331-332` | **Required**(消减合同+0.5mm 钳制边界的数值余量;§6 证明其产出是参考基准) | **保留** |
| cloud 重建算术 | `cloud.py:160-161` | **Required**(与上同源;条件 float32 还回 188-189) | **保留** |
| domain 统一 coerce | `_validation.py:43` | **Required**(统计合同,测试以 np.float64 精确断言,`test_serving_selection_shape.py:210`) | **保留** |
| ensemble 统计/pdf/wind stats | `statistics.py`、`pdf.py`、`wind.py:221-506` | Required(同上合同) | 保留 |
| tile 窗口升 f64 | `tiles.py:718-720` 等 10+ 处 | Probably unnecessary for correctness(色带映射 f32 足够)但**非本次迁移范围**,改动只增加回归面 | 保留不动 |
| 2×2 插值窗口 | `point_forecast.py:1856`(legacy 路径) | Probably unnecessary | 保留 |
| 坐标/几何 | `tiles.py:629-659`、grid math | Non-impacting(标量/元数据量级) | 保留 |
| 类型注失真 | `pipeline.py:252` 注 float64 实为 float32 | 文档性错误 | 建议顺手修注释 |
| JSON 展开 | `point_forecast.py:603` float32→float64 文本 | Non-impacting 但见 §19 | 见 §21-2 |

未发现可安全删除的 float64(删除收益≈0,回归风险>0)。

---

## 16. Storage Format Versioning(推荐兼容架构)

**推荐:Option B —— `sharded_v2`,per-variable native dtype,manifest 携带版本。**

- `STORAGE_FORMAT_VERSION`(ingestion `config.py:236`,现 `"sharded_v1"`)→ 写 `"sharded_v2"` 于 manifest(`coordinator.py:987,1256`)。**无需任何 DB 迁移**(catalog 无 format 列;per-cycle store 隔离;`model_runs` 不可变 store path,`catalog.py` immutable)。
- v2 语义:**连续量 f16(little-endian `<f2`,显式端序,修复 v1 的 native 端序含糊)、flags u8(det/mem shard)、mean-shard flags 保持 f32(成员均值概率 0..1)、NaN 哨兵不变**;index/trailer/对象键/生命周期完全不变(几何与 dtype 无关,`zarr.py:406` 运行时取 offset)。
- reader:`manifest_storage_format()` 分支已有(5 个调用点);`ShardedV1Reader.read_chunk` 按 format 选 frombuffer dtype 并**统一物化为内部 dtype**(point/window 立即 f32/f64 域;chunk cache 可保 f16——见 §10)。建议新 reader 类 `ShardedV2Reader` 复用 `ShardedV1Reader` 的 index/cache 机制,而不是在 v1 类内加 if——v1 路径冻结,降低回归面。
- **旧 cycle 保持 v1 可读、新 cycle 写 v2、旧数据随 lifecycle 自然淘汰——零 bulk migration**:canonical 选择只读"最新可服务 cycle",v1 cycle 在其活跃窗口内由 v1 分支服务,直到 GC 回收(`planner.py`;reclamation 队列按 `physical_key` 与 `store_generation` 基线防护,`worker.py:184-232`)。
- 前提约束:同一 cycle 的首次写与 finalization 之间**不得切换** `STORAGE_FORMAT_VERSION`(`coordinator.py:1256` per-lead 都盖章;manifest 30s 正向缓存,`manifest_reader.py:39-43`)。

---

## 17. NaN / Inf / Fill / Missing Semantics

- 平台**无 `_FillValue`**;缺失 = index length==0 → NaN fill chunk(`zarr.py:407-408`);writer NaN-fill 缓冲(`zarr_writer.py:658`);有效性一律 `np.isfinite` mask;domain 拒绝非有限输入(`_validation.py:55`);NaN→JSON null(`point_forecast.py:554`)。
- **f16 往返实测**:NaN 保真(bytes 0x007e)、±Inf 保真、±0.0 符号保真;**70000.0 → inf(RuntimeWarning overflow in cast)、65519→65504(饱和)、<6e-8 下溢为 0、6e-6 为亚正规保真**。
- 对当前变量清单的溢出审计:实测 max(306.4mm、230.8km/h、±61.5m/s、24.1km、2.38m、100%、39.2°C)**全部 ≪ 65504**;PRATE 83.9mm/h、ceiling 20km 安全。**但 f16 范围检查应作为 writer 侧断言**(上游异常值/单位错误时 astype 会静默产生 inf,而 `np.isfinite` 有效性检查会把 inf 当无效丢弃——语义可保,但应在 encode 处 warning,防止静默面扩大)。
- 搜索 4-byte/byte-pattern 假设:仅 `zarr.py:421`(frombuffer f32,v2 修改点)、`png.py:63,76,99`(RGBA 4B/px,无关)、`wind.py:722`(int16 vector payload ×2,无关)、`client.ts:436-442`(vector header getFloat32,无关)。index/trailer 数学 dtype 无关。

---

## 18. Categorical Variables(单独分析)

- 归一化后 uint8(`pipeline.py:292-298`);lead-0 uint8 零(`pipeline.py:488-499`);**但 sharded writer 把它们并回 f32 存储**(`zarr_writer.py:658`),而 `.zarray` 元数据记 uint8(`zarr_writer.py:373`)——**元数据与字节不一致**。
- 还有第二条漏洞:若 GRIB units token 恰为 `"flag"`,单位换算短路的值**保持 float32**(`pipeline.py:766-767` 短路逻辑)——归一化 dtype 依赖上游属性,不稳定。
- **量化(实测,f32→u8,Zstd-5)**:crain −31.7%、csnow −26.6%、cfrzr −9.0%、cicep −5.2%,四 shard 合计 **−29.6%**;raw 减 75%。
- **例外——GEFS mean shard 的 flags 是成员均值概率**(0..1 float32,如 0.65,`test_precipitation_phase_companions.py:268`;相态产品消费分数 flags,`point_forecast.py:899` ≥0.5 判定)。mean shard 的 flags **必须保持 float**(f16 亦可:0..1 区间半 ulp ≤0.00049,4dp 输出取整后无差异;保守起见 v2 首版 mean-shard flags 可留 f32)。
- serving 输出 `round(float(x), 4)`(`point_forecast.py:575`)+ 前端 `>= 0.5` 判定(`FE/lib/forecast/precipitation.ts`)——u8/det 分支存 0/1 完全保真。

---

## 19. Redis / JSON / Frontend

- **Redis**(`cache.py:199`):`model_dump_json()` 的 **JSON 文本**;float16 只改变文本里的十进制字面量,缓存机制、key(sha256 of canonical JSON)、TTL、损坏检测(schema validate)全部 dtype 无关。**不存在"float16→Redis 自动减半"**。vector L2 是 int16 量化原始字节(`vector_field.py:198`),与存储 dtype 完全隔离。
- **JSON(Pydantic)**:schemas 全部 `float | None`;转换点全是 `float(...)`(无 `.item()`)。**两类字段**:
  1. **API 边界已舍入**(dtype 无关):direction/cloud 1dp、flags/概率/coverage 4dp、consensus 2dp 等(agent 4 §7 表);
  2. **未舍入直通**:`point_forecast.py:603-606` 的 temperature_2m/precipitation_rate/RH/wind_gust/visibility/snow_depth/cloud_ceiling 以 float32→float64 **展开文本**输出(现状就是 `15.100000381469727` 这类噪声尾数)。f16 存储后这些字面量变为 f16 展开(如 `15.1015625`)。**JSON 语义不变、字节文本变**;对 `Number.isFinite` 门控与 Intl 渲染无影响,但任何对响应做黄金文件断言的测试/客户端会看到差异。
- **Frontend**:fetch JSON(`client.ts:71,86`);所有小数处理是渲染期格式化(toFixed/Intl,`labels.ts:86-117`);客户端统计复刻 ddof=0/linear percentile(`transform.ts:273-301`);flags `>=0.5`、dry `<=0.05`、静风 `hypot<0.5`(int16 量化域,`windParticles.ts:14`)——**这些阈值比较发生在 JSON Number 上,与存储 dtype 无关**(§21-2 的显示舍入项除外)。vector 二进制解码(`client.ts:404-478`)与 f16 无关。**不存在"frontend 内存自动减半"。**
- **presentation rounding 与 storage precision 已天然分离**(API round + FE Intl);本轮**不引入任何全局 round/truncate**——§21-2 是唯一需要产品决策的显示层项,且建议保持现状。

---

## 20. Backward-Compatible Rollout Plan(只设计,不实施)

**Phase 0 — 前置硬化(1 个小 PR 量级)**
- writer 侧 f16 范围断言 + encode 前非有限检查(§17);
- flags 归一化短路漏洞修复(`pipeline.py:766` 强制 u8);
- (可选)钳制边界补偿参数化(§6-4)。

**Phase 1 — dual reader(先读)**
- 新增 `ShardedV2Reader`(或 v1 类的 dtype 表参数),`manifest_storage_format()` 分支全覆盖 5 个调用点;v1 路径零改动;
- chunk cache 按 manifest dtype 物化(连续量 f16 缓存);
- 测试:v1 fixture 全绿 + v2 fixture 等价断言(§18 测试计划)。

**Phase 2 — dual writer(后写)**
- `STORAGE_FORMAT_VERSION=sharded_v2` 灰度:先 GFS(det,风险最低)→ GEFS mean → GEFS members;
- 同一 cycle 内禁止切换(启动时快照设置);
- writer 集成 `zarr_writer.py:658` 的 dtype 表(f16/u8/f32-mean-flags)。

**Phase 3 — mixed serving 验证**
- 生产影子比对(§19):同一请求走 v1 旧 cycle 与 v2 新 cycle,逐变量 diff 阈值(§18 表的实测上界);
- 监控:per-format shard 计数、Range GET 字节、API RSS、clamped/invalidated 计数(现有 QC 日志 `pipeline.py:355-368`)。

**Phase 4 — 自然退役**
- v1 cycle 随 canonical 淘汰被 GC 回收(`planner.py`,reclamation `store_generation` 基线防护);无 bulk migration、无格式转换作业;
- v1 读分支**永久保留**(历史 tombstone/cycle 可能长寿)或按团队政策在 N 个 release 后移除。

**回滚**:Phase 2 前零风险;Phase 2 后把 env 切回 `sharded_v1` 即停止产生 v2(已写 v2 cycle 依靠 Phase 1 reader 继续服务);无需数据回写。

---

## 21. Required Tests / 19. Real-Data Validation(合并为验证计划)

**Unit**(新增):
- f16/u8/f32 serialize round-trip(含 NaN、±Inf 拒绝策略、±0.0、65504 饱和、下溢);v2 index/trailer 不变性;manifest `sharded_v2` 读写;mixed-dtype reader(v1+v2 同进程)。
- **Numerical regression(逐变量,用本报告的实测上界作断言)**:对每个连续变量,f16 往返误差 ≤ §5 表 max 值 ×1.2。
- **Precipitation**:正常 deaccum;极小增量(1 量子);大累积+小增量(64/128/256mm 三档);reset 边界(lead 3/6/9);负残差三段(≥0 / 钳制 / NaN);**前驱 f16 存储**(fallback 路径)+ **前驱内存 f32**(生产路径)两条;对抗 AC1-AC5 复现(1e-6 级事件只允许在钳制边界,且不得批量)。
- **Cloud**:reconstruction ±5pp 三段;f16 前驱传播 ≤0.1pp;tolerance 边界值。
- **Ensemble**:mean/median/std/P10/P25/P75/P90/IQR 对 f32 基线 max ≤0.03°C(30 成员);phase support uint8 不变性;joint amount-phase 阈值翻转率 0。
- **Interpolation**:2×2 解析场 f32 vs f16 输入 max ≤ 半 ulp×1.2;`test_serving_selection_shape.py:210` 的 float64 精确合同**仅对 v1 保持**,v2 定义新合同。
- **Mixed-format**:v1 cycle + v2 cycle 同 request 混合服务(cross-cycle fallback 路径 `test_cross_cycle*.py` 扩展)。
- **合同更新(必须同步)**:float32 fixtures(`test_serving_chunk_equivalence.py:18-27` 等)、`abs=1e-9` 位等价(改为 v1-only 或 per-format)、`abs=1e-5` deaccum/插值容差(v2 放宽到量化步长 + 1e-5)。

**Real-data validation(plan,用本报告管线)**:对 1 个完整 GFS + 1 个完整 GEFS cycle 逐变量/逐 lead 输出 max/MAE/P50/P95/P99 绝对误差与 max 相对误差;GEFS 产品单独输出 mean/median/std/P10/P25/P50/P75/P90/IQR/phase-support%;检查 threshold/classification/member-ordering 变化;**worst-case 样本表**(variable/cycle/lead/member/grid index/baseline/candidate/下游差)——本报告 §5/§6/§8 的实测表即该管线的首次运行结果,实施前应在目标生产硬件上复跑。

---

## 20. Quantified Expected Benefits(实测支撑汇总)

| 维度 | raw theoretical | post-Zstd 实测 | whole-process 实际 |
|---|---|---|---|
| 对象存储 | −50%(payload) | **−22.2%**(GFS 0.83→0.64GB/cycle;GEFS 23.7→18.4GB/cycle;活跃 ~507→394GB) | 同左(对象存储即最终态) |
| Range GET 字节 | −50% | **−18~19%**/请求;请求数 −0% | 冷延迟:二阶改善(传输+解压),非 50% |
| API chunk 缓存 payload | −50% | −50%(19.58→9.83 MiB/reader) | reader 总缓存 **−10~15%**;每 worker −78~95 MiB(8 readers) |
| ingestion RSS | ~0(解码 f32 保持) | 编码缓冲 −50%(2.3MB/shard) | **<2-3%** |
| chunk 解码 CPU | — | frombuffer+copy **2× 快** | serving 热路径小增益 |
| writer CPU | — | astype f32→f16 +10.9ms/4M(~370M/s) | ingestion 编码可忽略增量 |
| 网络出口(MinIO→API) | −50% | ~−20% | 每天按请求量折算 |

---

## 22. Recommended Target Numerical Policy

```yaml
decode / normalize / derived (deaccum, cloud):
  float32 存储, float64 仅在两个派生函数内部(现状保留)
continuous persistent forecast fields (v2 shards):
  float16 (little-endian <f2)
categorical fields (det/mem shards):
  uint8 (native; 修复现被并回 f32 的不一致)
ensemble mean-shard flags:
  float32 (成员均值概率 0..1;可后续降 f16)
decompressed API chunk cache:
  float16 / uint8 (native dtype, 按 manifest)
interpolation / ensemble statistics / reductions / tile color mapping:
  float32 域起算(domain 合同实际 float64,保持)
float64:
  仅现状已证明必要处(pipeline.py:331-332, cloud.py:160-161, _validation.py:43)
JSON / presentation:
  与存储 dtype 无关;不新增 round;保留现有 round 合同
float16 arithmetic:
  禁止(实测 8.7×慢 + reduction 溢出)
```

## 23. Go / No-Go Preconditions

1. 产品方接受 1dp 显示值在 f16 量化边界翻转(温度最多 ~25% 格点差 0.1°C,均方差 ≤0.027°C)与未舍入 JSON 字面量文本变化(§19)——**real blocker(产品决策)**。
2. f16 范围/非有限 writer 断言落地(§17)——**real blocker(防静默 inf)**。
3. flags u8 化 + mean-shard flags 例外 + `pipeline.py:766` 短路修复——**implementation detail**。
4. 测试合同同步(float32 fixtures、1e-9 等价、1e-5 容差按 format 分离)——**implementation detail,工作量大头**。
5. 钳制边界 −0.5 与 trace 0.10 的 1e-6 级 corner case:接受或参数化补偿——**optional optimization**。
6. 双 reader/双 writer rollout 按 §20 分四阶段,GFS 先行——**implementation detail**。
7. 文档修正(顺手):README GEFS "0.5°" vs 实现 pgrb2sp25 0.25°(`connector.py:186-188`);ARCHITECTURE.md:320 缓存尺寸;finalizer.py:20 "14-day" vs 实际 1 天保留;connector.py:71-72 "geavg out of scope" vs 实际摄入 geavg。

---

## 24. Three Architectures Compared

| 维度 | A 现状(f32+f32+Zstd) | **B 推荐(f32 计算 + f16/u8 存储 + f16 缓存)** | C f16 everywhere |
|---|---|---|---|
| 存储 | — | **−22.2%**(实测) | ≈B(flags 反而更差) |
| 网络/GET | — | **−18~19% 字节**,请求数不变 | ≈B |
| API 缓存 | — | payload −50%,reader 总 −10~15% | 同 B |
| CPU | 基线 | 解码 2× 快,编码 +小成本,**计算不变** | **算术 8.7× 慢 + reduction 溢出(实测)** |
| 精度 | bit 基线 | 阈值翻转 0(实测),max 0.027°C/0.0625mm | **破坏性**(600mm 处小增量 100% 丢失,实测) |
| 复杂度 | — | 中(新 format + dtype 表 + 测试合同) | 低(看似)但需重写全部数值安全网 |
| 兼容/风险 | — | manifest 机制现成,零 DB 迁移,可回滚 | 不可回滚的语义破坏 |

**未考虑第四架构(如 bfloat16/zfp/bitshuffle)的理由**:bfloat16 精度(8 位尾数)劣于 f16 且生态支持差;zfp/有损压缩改变"逐值可解释"合同且是更大变更;bitshuffle/byte-plane 重排是正交的压缩优化,可在 v2 之上独立评估。

## 25. Decision Criteria(逐条对照产品假设)

小于 1°C 温度量化 ✅(0.027°C);RH/云 ✅(0.03%);风/能见度/云底 ✅(0.014m/s / 7.8m / 0.0078km);降水存储误差 ✅(≤0.125mm 且 ≤1 上游打包量子);forecast uncertainty ≫ storage error ✅(GFS 2m 温度预报误差量级 1-3°C);不改产品语义 ✅(全部阈值翻转率 0,除 1e-6 对抗 corner);deaccum/reset 正确 ✅(§6);ensemble materially equivalent ✅(§8);吞吐/延迟无明显 regression ✅(§12/§14:编码 +小成本,解码更快,GET 字节降);backward compatible ✅(§16/§20)。

## 26. Risks / Blocking Questions

**Real blockers**:① 未舍入 JSON 字面量变化的产品接受度(唯一面向用户的文本差异);② 上游极端值 >65504 的静默 inf(必须 writer 断言)。**Implementation details**:测试合同迁移(工作量最大)、flags 元数据/字节不一致修复、v2 reader 类边界、`STORAGE_FORMAT_VERSION` per-cycle 冻结。**Optional**:钳制边界补偿、visibility/snow 之外字段的 byte-plane 重排、mean-shard flags 降 f16。

---

### 附:证据与可复现性
- 代码证据:全部 file:line 由 4 个独立调查 agent 交叉核对(ingestion 链路 / 存储与 serving / 变量与 GEFS / 全仓 dtype),关键结论由主调查者直接读码复核(`pipeline.py:302-420`、`cloud.py:103-235`、`zarr_writer.py:615-672`、`zarr.py:316-532` 等)。
- 数据与基准:85 个真实 cfgrib 解码字段(GFS 20260923/18z f003/f006/f189/f192 + GEFS 20260923/00z 全部 30 成员 f120,0.5° pgrb2a 代理 0.25° pgrb2s);基准脚本 fetch/decode/bench1-5 位于 `%TEMP%\f16bench\`,结果 JSON 在 `results\`;CPU=AMD Ryzen 7 5800X,numpy 2.5.3,numcodecs Zstd(level=5)。
- 已知偏差:GEFS 成员基准用 0.5° 文件(0.25° 平滑度略低→压缩比估计保守);cloud_ceiling 未采样,压缩估计用 cloud 类比;活跃保留窗口由 canonical 淘汰机制推导(GFS ~40 报/GEFS ~20 报),非硬配置。

---

## Addendum (post quantization benchmark round)

Follow-up benchmark ([`QUANTIZATION_BENCHMARK.md`](./QUANTIZATION_BENCHMARK.md)) refined two conclusions of this report:

1. **§5 GEFS trace-threshold flips**: the "zero flips" claim holds in float32 comparison semantics, but the production path compares after float64 promotion (`precipitation.py:184` `float(amount_curr)`). GEFS decimal packing produces a large mass of values exactly equal to `f32(0.1)` (12.813% of the f120 precipitation field); under plain float16 storage these flip wet→dry in phase products. `precipitation_amount_3h` therefore keeps **float32** storage in the recommended `sharded_v2` freeze (see QUANTIZATION_BENCHMARK.md §26).
2. **Compression**: plain float16 remains the recommended representation; variable-specific/fixed-point quantization variants were measured at ≤+3.6% relative on cycle-weighted totals and are rejected on complexity grounds.
