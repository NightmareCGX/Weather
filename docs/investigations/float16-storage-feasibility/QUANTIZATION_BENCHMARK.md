# Weather Platform Quantization / Float16 / Zstd Storage Benchmark Report

**性质**:Storage Representation / Quantization Benchmark(follow-up to [`REPORT.md`](./REPORT.md))。只做基准,不改 production code,不重新调查已确认的数值架构。所有候选共用同一批真实 float32 源场、同一 100×100 chunk 几何、同一 numcodecs Zstd **level 5**;compute-domain dtype 一律不变。

**最终结论(先行):冻结 `sharded_v2` 为 Option A —— 连续量 plain float16 + det/mem flags uint8(mean-shard flags 保持 float32),但 `precipitation_amount_3h` 例外存 float32(阈值语义,非压缩原因)。全周期加权节省 ≈ −22.0%;plain f16 已实质处于 Pareto 前沿,任何 quantized-f16 变体在全周期加权下增益 ≤3.6%(相对),不值得永久策略成本。**

---

## 1. Executive Summary

- **16 个存储表示方案 × 21 个真实字段实例(15 GFS + 6 GEFS)× 真实 chunk 几何实测**。实例加权与全周期加权( GEFS 占 96.6% 字节)结果分化显著——多数方案的"优势"在实例加权下被放大。
- **f32 容器内的任何 quantization 都无法接近 float16**:uniform round3/2/1 仅省 3.9/9.3/21.0%;variable-specific(conservative/balanced)3.2/10.6%;即使把 mantissa 修剪到 **f16 同等精度(11 effective bits)** 也只省 8.3%。→ f16 的压缩优势主要来自 **2 字节容器本身**,不是尾数熵(§10/§17,Q8/Q9)。
- **round+f16 的"进一步压缩"是幻觉居多**:round3+f16 与 plain f16 的 f16 码本几乎相同(温度/RH/GEFS 降水 unique codes 完全不变),仅 +1.0pp;round2+f16 实例加权 +4.5pp,但**全周期加权只剩 +3.6%(相对)**,且收益集中在 snow(+62%)/rate(+30%)/vis(+11%)三个小字段,在四个大字段(temp/RH/cloud/wind)上 ≈0,在 GFS 降水上**倒退 3.6%**。
- **fixed-point int16/uint16 在全周期加权下对 plain f16 增益 +0.1%**——GEFS 侧 RH/temp 的损失抵消 GFS rate/vis 的收益。按复杂度原则直接否决(Q10)。
- **发现一个真实阈值地雷(本轮最重要发现)**:GEFS 十进制打包产生**恰好等于 f32(0.1) 的 0.100mm 格点,占该场 12.813%**。生产代码在 float64 域比较(`precipitation.py:184` `float(amount_curr) <= 0.10`)→ 基线判**湿**;plain f16 把 0.1000000015 舍入到 0.0999755859375 → 判**干**。即 plain f16 会使 GEFS 毛毛雨区 12.8% 的格点在相态产品中湿→干翻转。保十进制网格的方案(round2/fixed-point 0.01)精确保留该值,零翻转。→ 驱动 `precipitation_amount_3h` 的 dtype 例外(§14/§26)。
- **推荐**:见 §26。收益从上一轮报告的 −22.4% 微调至 **−22.0%**(为降水阈值安全放弃降水的 f16 化,≈ −0.4pp)。

## 2. Scope and Existing Evidence

不重新调查(继承 REPORT.md):float16 算术已否决(8.7× 慢 + reduction 溢出);Zstd level 不是主要杠杆(L12 仅 +3.3%,L19 +9.9%,均远低于 f16 的 22%);de-accumulation/cloud 重建的 float64 计算合同;API 缓存结构;rollout 机制。本轮只回答:**`sharded_v2` 的 continuous payload 应该是什么表示**。

## 3. Dataset and Methodology

- **数据**:复用上一轮 85 个真实解码字段中的 21 个实例(GFS 20260923/18z 0.25°:temp/RH/u/v/gust/prate/precip f006+f192/precip f189/cloud/vis/snow/4 flags;GEFS 20260923/00z gep01 f120:temp/RH/u/v/precip/cloud)。cloud_ceiling:**not sampled**(周期估算用 cloud 类比,标注 est)。
- **几何**:100×100 chunk(721×1440 → 8×15=120 chunks,边缘 chunk 不填充,同 writer 语义)。
- **压缩**:numcodecs `Zstd(level=5)`,每 chunk 独立压缩(与 `zarr_writer.py:628,658-660` 一致)。
- **误差基准**:dequantize→float32 vs 原始 float32 源;相对误差仅报 |x|≥1.0(internal unit)。
- **阈值比较域说明**:bench6 自动翻转检测在 float32 域执行(NEP 50 weak scalar);生产代码经 `float()` 提升在 float64 域比较。两者在"恰好位于阈值上的值"上结论不同——本报告对这类值做了逐域复核(§14),这是本轮方法论上最重要的修正。
- 硬件:AMD Ryzen 7 5800X,numpy 2.5.3,Python 3.14.6。脚本 `bench6_quantization.py`/`bench7_cpu_cycle.py`,结果 `results/bench6_quantization.json`/`bench7_cpu_cycle.json`。

## 4. Current Float32 Baseline(Scheme A)

15 个连续量实例合计:raw 44.56 MB → compressed **10.78 MB**(平均比率 4.13×)。Per-variable compressed KB 见 §16 表。Mean chunk:温度 8015B、RH 11475B、风 9857B、降水 4456B。这是所有 saving 的分母。

## 5. Plain Float16 Baseline(Scheme B)

实例加权 **−22.0%**(10.78→8.41MB);全周期加权 **−22.4%**。max err 0.125(仅 GFS 降水 306mm 处,f16 spacing);**f32 域阈值翻转:全部为零**;float64 生产域复核:唯一翻转类 = GEFS 恰好-0.1mm(§14)。API chunk 缓存 payload 40,000→20,000B。

## 6. Uniform Decimal Quantization + Float32(Scheme C)

| | saving vs A(实例) | max err | 关键翻转(f32 域) | 判定 |
|---|---|---|---|---|
| C1 round(3) | 3.9% | 5.0e-4 | vis fog 13 格(0.0013%) | **dominated**(比 B 差 23.2pp,RAM 2×) |
| C2 round(2) | 9.3% | 5.0e-3 | prate-info 0.41%;vis fog 0.0087% | **dominated** |
| C3 round(1) | 21.0% | 0.05 | **trace 7.10%**(0.125→0.1);freeze 0.18%;prate 2.87% | **dominated + 翻转不合格** |

Q1:**round(3)+f32 比 plain f16 大 23.2%**(10.36 vs 8.41MB),更大而非更小/接近。Q2:round(2)+f32(9.3%)**远不能**达到 f16(22.0%),且 RAM 仍 2×,无资格。

**机制发现(适用于所有十进制网格 × GRIB 二进制打包)**:GFS 降水 0.125mm 网格值在 0.1 阈值附近的行为——round(1) 把 0.125 → 0.1 → `<=0.10` 判干,**7.1% 全场格点湿→干翻转**。任何与 0.10/0.125 网格耦合的十进制量子都有此病理。

## 7. Uniform Decimal Quantization + Float16(Scheme D)

| | saving vs A | vs B | f16 码本变化 | 判定 |
|---|---|---|---|---|
| D1 round(3)+f16 | 22.7% | **+1.0pp** | 温度/RH/降水 unique codes **完全不变**(值改变 5-27% 但落回同码) | **dominated by B**(为 1pp 引入策略) |
| D2 round(2)+f16 | 26.5% | +4.5pp | vis 6843→2120 codes;snow/rate 大幅;temp/RH/云 ≈0 | **前沿候选,全周期仅 +3.6%**(见 §16/§21-23) |
| D3 round(1)+f16 | 36.6% | +18.7pp | — | **不合格**:trace 7.1% 翻转(0.125→0.1) |

Q4:round3+f16 **不**能进一步显著减少 compressed size(熵分析见 §17);round2+f16 可以但集中;round1+f16 数字最好但语义破坏。

## 8. Variable-Specific Quantization + Float32(Scheme E)

E1 conservative(温度 0.01°C/RH 0.05%/风 0.01m/s/vis 0.001km/降水 0.01mm…):**3.2%** — 精度优于 f16 但压缩可忽略,**dominated**。
E2 balanced(0.05/0.1/0.05/0.01/0.05…):**10.6%**,且降水 0.05mm 量子与 0.125 GRIB 网格产生系统 tie(0.125/0.05=2.5→banker's→2→0.1)→ **trace 7.1% 翻转,不合格**。

Q3:**不能**。f32 容器下即使 variable-specific 量化,压缩(≤10.6%)与 f16(22.0%)差距悬殊,且 RAM 2×——精度优势对产品无意义(G11 已证同等精度也只值 8.3%)。

## 9. Variable-Specific Quantization + Float16(Scheme F)

F1 conservative+f16:**23.3%**(+1.3pp vs B 实例;全周期 ≈+0.4%)→ **dominated by B**。
F2 balanced+f16:实例 28.1%(+6.1pp)——但继承 E2 的降水 0.05 病理 → **trace 7.1% 翻转,不合格**。若把 F2 的降水量子改为 0.01mm(即 D2 混合),则退化为 D2 + 无收益的 per-var 复杂度。

Q5:**variable-specific quantization+f16 不构成 plain f16 的真正 Pareto 改进**:唯一存活的变体(D2 形态)在全周期加权下 +3.6%,却引入永久 per-variable 量子表、阈值感知测试、显示翻转面翻倍(§15)。

## 10. Binary Mantissa Grooming(Scheme G)

frexp/ldexp RTN,保持 NaN/±Inf/±0、float32 容器与指数域:

| effective significand bits | saving vs A | max err | 备注 |
|---|---|---|---|
| 17 | 4.5% | 0.0019 | exactly-0.1 保持(向上舍入) |
| 15 | 5.1% | 0.0075 | 恰好无翻转(网格对齐侥幸) |
| 13 | 5.8% | 0.0625 | |
| **11(≈f16 精度)** | **8.3%** | 0.1875 | encode CPU 50.5ms/4M = 4.7× astype |

Q8:**即使删除到 f16 同等尾数精度,4-byte 容器仍远输 2-byte f16**(8.3% vs 22.0%)。Q9(基于实验的分解):f16 的 22pp 优势中,**约 8pp 来自精度损失(尾数熵消除),约 14pp 来自 raw 字宽减半**。低尾位熵在 f32 容器内不是 Zstd 的主要瓶颈——容器布局才是。附带否决理由:frexp/ldexp 管线 CPU 4.7×,收益却不比 C 系列好多少。

## 11. Fixed-Point / Scale-Offset Packing(Scheme H)

Per-variable u16/i16 + NaN sentinel(温度 i16 0.01°C;RH/cloud/gust/prate/precip/vis u16 0.01;snow u16 0.001;风 i16 0.01);范围断言全通过(实测 max:temp −68..39°C、precip 306mm、gust 231km/h、风 ±62m/s)。

- 实例加权 25.4%(vs B +4.3pp);**全周期加权 −22.3% vs A,对 B 仅 +0.1%**——GEFS 侧 RH(−9~10% 对 B)/temp 的损失抵消 GFS rate(+35%)/vis(+14%)/GEFS 风(+11%)的收益。
- 误差特征最佳(小值域 max 0.005mm/0.0023°C;2dp 显示几乎不翻 1.5%)。
- Q10:**否决**。全周期零增益 + scale/sentinel/metadata/range/overflow/dequant 永久机器成本。§27 的"3-5% → 否决"原则以其最强形式命中(0.1%)。

## 12. Categorical uint8

det/mem flags:唯一合理候选,`uint8 → Zstd L5`(与 REPORT.md §18 一致)。bench6 复测:4 个 flag 场 round-trip 零 mismatch;总压缩 91.5KB vs 现状 f32 133.1KB(−29.6%,bench4 口径)。**不做任何 float quantization 变体**。GEFS mean-shard flags 是成员均值概率(0..1),保持 float32,不在本轮比较范围(not applicable)。

## 13. Per-Variable Numerical Error(serious candidates)

max / MAE / P99(internal units;完整 JSON 在 results/):

| field | B plain f16 | D2 r2+f16 | H fixed |
|---|---|---|---|
| gfs/temperature (°C) | 0.027 / .0031 / .0148 | 0.027 / .0037 / .0148 | 0.0023 / .0023 / .0023 |
| gfs/visibility (km) | 0.0078 / .0044 / .0066 | 0.0123 / .0048 / .0080 | 0.0048 / .0042 / .0048 |
| gfs/precip (mm) | 0.125 / 1.2e-6 / 0 | 0.125 / .0015 / .0059 | 0.005 / .0015 / .0050 |
| gefs/precip (mm) | 0.050 / 1.0e-4 / .0016 | 0.050 / .0001 / .0016 | 7.6e-6 / 1.7e-8 / 0 |
| gfs/snow (m) | 0.00097 / 3.0e-5 | 0.00097 / ~同 | 0.0005 |

所有候选的 max/typical 误差都远小于显示粒度与 REPORT.md §5 的产品尺度;误差不构成区分度,**阈值行为才是**(§14)。

## 14. Threshold / Semantic Stability

f32 域自动检测(bench6)+ **float64 生产域逐例复核**(`precipitation.py:184` 语义):

| 方案 | f32 域翻转 | float64 生产域复核 | 判定 |
|---|---|---|---|
| **B plain f16** | 全零 | **GEFS exactly-0.1mm:12.813% 场格点 湿→干**(0.1000000015→0.0999755);其余全零 | 唯一语义议题,见下 |
| C1/D1/F1 | vis fog 13 格 | 同 | 可忽略 |
| C2/D2/H | prate-info 0.41%*;vis fog 0.0087%(0.9952→1.0) | exactly-0.1 **保持湿**(十进制网格含 f32(0.1)) | fog 翻转 90 格/场,微小但真实 |
| C3/D3/E2/F2 | **trace 7.10%/6.22%**(0.125→0.1);freeze 0.18% | 同 | **不合格** |
| G15 | 零 | exactly-0.1 保持(侥幸对齐) | 脆弱 |
| G17/G13 | 零 | exactly-0.1 保持(向上舍入→湿) | |

*prate 0.1mm/h 是本轮的信息性阈值,非 code-verified 产品阈值。

**exactly-0.1mm 地雷的完整机制**(本轮核心发现):
1. GEFS 十进制打包产生海量恰好 0.100mm 值(实测该场 12.813%, widespread drizzle);GFS 二进制 0.125 打包**无**此值(0.125≠0.1)→ 仅 GEFS 受影响。
2. 基线(f32 存储)下,serve 路径 `float(np.float32(0.1)) = 0.10000000149 > 0.10` → **判湿**——这其实是 float 展开噪声对政策意图(`≤0.10 → dry`)的偏离。
3. plain f16 舍入到 0.0999755859375 → **判干**——恰好回到政策意图,但**改变了今天的相态产品行为**(drizzle 区 phase-support/transition 统计 12.8% 格点位移)。
4. 保十进制网格的方案(round2 / fixed-point 0.01)把 0.1000000015 精确保留为 f32(0.1) → 零翻转。
5. 这同时修正上一轮 REPORT.md §5/§6 的一个结论:那里"GEFS trace 翻转 = 0"在 float32 比较域成立,但在生产的 float64 域不成立(见 REPORT.md 末尾 addendum)。

**处置选项**(§26 给出推荐):*(i)* plain f16 + 接受"政策意图修正"型翻转(需产品 sign-off);*(ii)* 降水单独 round(0.01mm)+f16(保基线,引入单变量量子);*(iii)* **降水单独保持 float32**(保基线,零机器,v2 本就 per-variable dtype;放弃降水的 f16 压缩 ≈ 全周期 +0.4pp)。

## 15. Presentation-Level Effects

| 方案 | max 1dp 翻转 | max 2dp 翻转 | 备注 |
|---|---|---|---|
| B | 24.8%(temp) | 76.7% | 与 REPORT.md §5 一致 |
| D2 | **50.3%** | 76.7% | round2 网格使 1dp 边界翻转翻倍 |
| H | 52.5% | **1.5%** | 0.01 网格 = 2dp 网格,2dp 几乎不变 |

presentation flip ≠ semantic failure(REPORT.md §19 已分离);但 D2 把 1dp 显示翻转面扩大 2 倍,是用户可见 diff 面的实质扩大。

## 16. Compression Results

**Per-variable compressed KB(实例,Zstd L5)**:

| field | A | B | D2 | H | G11 |
|---|---|---|---|---|---|
| gfs/RH | 1345 | 1168 | 1168 | 1286 | 1284 |
| gfs/wind_u / v | 1155/1152 | 925/916 | 911/914 | 906/893 | 1126/1119 |
| gfs/gust | 1081 | 959 | 958 | 948 | 1160 |
| gfs/cloud | 1032* | 838* | 838* | 889* | — |
| gfs/temp | 939 | 763 | 759 | 776 | 931 |
| gfs/vis | 958 | 523 | 464 | 449 | 619 |
| gfs/prate | 796 | 622 | 436 | 404 | 744 |
| gfs/snow | 617 | 329 | **125** | 246 | 335 |
| gfs/precip | 522 | **424** | 440(−3.6%!) | 423 | 522 |
| gefs/wind_u / v | 576/575 | 453/443 | 422/417 | 400/394 | 493/497 |
| gefs/temp | 292 | 239 | 237 | 241 | 283 |
| gefs/RH | 371 | 317 | 316 | 346 | 346 |
| gefs/precip | 136 | 103 | 103 | 104 | 133 |

*cloud 为 bench6 补测值(120×8.7KB)。**关键行:GFS 降水 D2 比 B 差 3.6%**(round2 破坏 0.125 网格与 f16 的对齐)。

**统一 aggregate 表**(实例加权 saving vs A / 全周期加权 vs A / vs B):

| Scheme | Native | Quantization | Raw | saving vs current(实例) | (全周期) | vs plain f16(全周期) | API cache payload | max err | 阈值判定 | CPU(encode 侧) | Complexity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | f32 | none | 44.56MB | 0% | 0% | — | 40,000B | 0 | 基线 | 基线 | — |
| **Plain f16** | f16 | binary16 | 22.28MB | **22.0%** | **−22.4%** | 0 | **20,000B** | 0.125 | exactly-0.1 议题 | astype 10.8ms | v2 基线成本 |
| round3 f32 | f32 | 0.001 | 44.56MB | 3.9% | ~4% | **更差 23%** | 40,000B | 5e-4 | ok | 3.5ms | 低但 dominated |
| round2 f32 | f32 | 0.01 | 44.56MB | 9.3% | ~9% | 更差 16% | 40,000B | 5e-3 | ok | 3.5ms | dominated |
| round1 f32 | f32 | 0.1 | 44.56MB | 21.0% | ~21% | ≈但误差劣 | 40,000B | 0.05 | **trace 7.1%** | 3.5ms | 否决 |
| round3 f16 | f16 | 0.001+b16 | 22.28MB | 22.7% | ~22.5% | +0.1pp | 20,000B | 0.125 | ok | 14.5ms | dominated |
| round2 f16 | f16 | 0.01+b16 | 22.28MB | 26.5% | **−25.2%** | **+3.6%** | 20,000B | 0.125 | fog 90 格 | 14.5ms | +永久策略 |
| round1 f16 | f16 | 0.1+b16 | 22.28MB | 36.6% | ~36%* | +18pp* | 20,000B | 0.125 | **trace 7.1%** | 14.5ms | 否决 |
| var-cons f32 | f32 | per-var | 44.56MB | 3.2% | ~3% | 更差 25% | 40,000B | 0.021 | ok | 7.4ms | dominated |
| var-bal f32 | f32 | per-var | 44.56MB | 10.6% | ~10% | 更差 15% | 40,000B | 0.049 | **trace 7.1%** | 7.4ms | 否决 |
| var-cons f16 | f16 | per-var+b16 | 22.28MB | 23.3% | ~22.7% | +0.4pp | 20,000B | 0.125 | ok | 18.4ms | dominated |
| var-bal f16 | f16 | per-var+b16 | 22.28MB | 28.1% | ~25%* | +3.7pp* | 20,000B | 0.125 | **trace 7.1%** | 18.4ms | 否决 |
| groom 17/15/13 | f32 | binary | 44.56MB | 4.5/5.1/5.8% | ~同 | 更差 20%+ | 40,000B | ≤0.063 | ok(侥幸) | **50.5ms** | 否决 |
| groom 11 | f32 | ≈f16 精度 | 44.56MB | 8.3% | ~8% | 更差 17% | 40,000B | 0.1875 | ok | 50.5ms | 科学对照组 |
| fixed-point | i16/u16 | per-var scale | 22.28MB | 25.4% | **−22.3%** | **+0.1%** | 20,000B+dequant | 0.005 | ok | 12.0ms | 高机器,零增益 |
| **flags u8** | u8 | native | ÷4 | **−29.6%** | 同 | — | 10,000B | 0(零 mismatch) | 零 | 最低 | 已在 v2 计划内 |

\* 全周期值受 GEFS 权重修正,round1/var-bal 的实例优势会缩水;其翻转不合格与权重无关。

## 17. Quantization + Float16 Entropy Analysis

unique f16 codes(plain vs quantized)与值变化率:

| field | plain unique | D1 unique | D2 unique | D2 值变化率 | D2 compressed vs B |
|---|---|---|---|---|---|
| gfs/temperature | 1070 | 1070 | **1070** | 32.8% | −0.5%(≈0) |
| gfs/RH | 934 | 934 | 934 | **0.0%** | 0.0% |
| gfs/precip | 941 | 941 | 941 | 45.1% | **+3.6%(更差)** |
| gfs/visibility | 6843 | 5575 | **2120** | 15.2% | −11.4% |
| gfs/snow | — | — | — | — | **−62.1%** |
| gefs/precip | 399 | 399 | 399 | **0.0%** | 0.0% |

**直接回答**:round(3)+f16 与 plain f16 产生**几乎完全相同的 half-float 码本**(温度/RH/GEFS 降水 unique 数完全不变;值被重新标注但落回同码)→ **no additional entropy reduction**。round(2)+f16 的真实熵降低只发生在天然十进制粗粒度的场(vis/snow/rate);对温度/RH/云/GEFS 降水为零或负。semantic quantization 的"进一步 collapse"在 f16 的 11-bit significand 面前基本没有空间——f16 自身已经是一个粗量化器。

## 18. CPU Cost

4M 元素(每 10K chunk 除以 400):

| 操作 | 4M ms | /chunk ms | 备注 |
|---|---|---|---|
| f32→f16 astype(writer) | 10.8 | 0.027 | 基线量化成本 |
| round2→f16(writer) | 14.5 | 0.039 | +34% vs astype,绝对值可忽略 |
| var-quant→f16 | 18.4 | 0.046 | |
| groom(17bit) | 50.5 | 0.126 | frexp/ldexp 贵 4.7× |
| fixed-point pack | 12.0 | 0.030 | |
| f16→f32(reader) | 18.9 | 0.047 | vs Zstd decode ~0.3ms/chunk 可忽略 |
| i16→f32 dequant | 5.6 | 0.014 | |
| f32 identity(copy) | 2.3 | 0.006 | |

quantization CPU 不是任何候选的决定因素(groom 除外,其 4.7× 成本叠加零收益构成否决项之一)。不据此推断 whole-system latency。

## 19. API Memory Impact

按 native payload(100×100 chunk):

| 表示 | chunk cache payload/reader(512 chunks) | compute transient |
|---|---|---|
| A / C / E / G(f32 族) | 19.58 MiB | — |
| **B / D / F(全部 f16 族)** | **9.83 MiB** | — |
| H(打包 i16) | 9.83 MiB(若缓存 packed)+ 每次 compute 前 dequant 40,000B/块 transient 或物化 20,000B f32 | +dequant 路径 |

object-store compression 与 RAM 表示必须分开记账:G11(groom)证明"压缩接近 B"的方案(如有)RAM 仍 2×;反之 H 压缩不占优却要付 dequant 复杂度。f16 族是唯一同时优化两个账本的类。

## 20. Range GET Byte Impact(request count 不变)

mean compressed chunk bytes(A→B→D2):温度 8015→6513→6475;RH 11475→9967→9967;风 9857→7895→7777;降水 4456→**3622**→3752;云 8853→6987→6987;vis 8171→4464→3957;snow 5265→2808→1063;prate 6790→5306→3720。GEFS 统计请求(30×temp):280→229KB(B)。point 请求 −18~19%(B);D2 在 snow/vis/rate 上多 −10~30%,在降水上**多 +3.6%**。

## 21. GFS Whole-Cycle Estimate

15 vars × 81 leads(ceiling 用 cloud 类比,est;flags 用 bench4 f32/u8 实测):

| | per cycle | vs A |
|---|---|---|
| A | 0.83 GB | — |
| B(+u8 flags) | 0.65 GB | −22.4% |
| D2 | 0.61 GB | −26.6% |
| H | 0.63 GB | −24.2% |

## 22. GEFS Whole-Cycle Estimate

14 vars × 81 leads × 31 sets(30 members + 1 mean;成员/mean 统一近似,est):

| | per cycle | vs A |
|---|---|---|
| A | 23.87 GB | — |
| B | 18.52 GB | −22.4% |
| D2 | 17.87 GB | −25.1% |
| H | 18.56 GB | **−22.3%(≈零增益)** |

## 23. Active Storage Estimate(estimate,沿用 REPORT.md 口径:GFS ~40 报 + GEFS ~20 报,由 canonical 淘汰推导,非硬配置)

| | active | vs A |
|---|---|---|
| A | 511 GB | — |
| **B** | **396 GB** | **−22.4%** |
| D2 | 382 GB | −25.2%(每双周期多省 0.68GB) |
| H | 396 GB | −22.4%(+0.1%) |

**全周期加权是本轮的关键修正**:实例加权下 D2 领先 4.5pp,GEFS 加权后缩水到 3.6%(相对)/2.8pp(绝对);H 直接归零。

## 24. Pareto Frontier

**Dominated(被 B 完全支配:压缩更差或相等 + RAM 更大/相等 + 误差不更好 + CPU 更差 + 复杂度更高)**:C1、C2、E1、G17、G15、G13、G11、D1、F1、**H**。
**不合格(阈值翻转)**:C3、E2、D3、F2(0.125×十进制网格 tie 病理,trace 7.1%)。
**前沿(2 个)**:
- **B plain f16**:−22.4% 周期,20,000B RAM,唯一议题 = exactly-0.1(可用单变量 dtype 例外解决);
- **D2 round2+f16**:−25.2% 周期,20,000B RAM,代价 = 永久十进制量子策略 + fog 边界翻转 + 1dp 显示翻转翻倍 + GFS 降水局部倒退。

D2 vs B 的边际交易:每双周期 0.68GB(≈ 全周期 +2.8pp 绝对),换取一个永久策略及其全部测试面。按 §27 原则判定不值得。

## 25. Complexity / Maintenance Cost

- **B(plain f16)**:v2 机器(dtype dispatch、reader/writer、测试合同)= 任何 v2 都要付的基线成本;**无** per-variable 量化表;新增变量默认 f16,零决策。
- **D2**:额外需要 variable→quantum 表、新变量入表政策、阈值感知测试(0.10/1km/60km/h 边界 × 网格耦合)、量化文档、科学合同;"十进制网格 × GRIB 二进制打包"耦合已被 E2/F2/C3 证明是系统性雷区;1dp 显示 diff 面翻倍。
- **H**:再加 scale/offset/sentinel/range/overflow/dequant/metadata;且全周期增益 +0.1% → 纯负期权。
- **G**:frexp 管线 4.7× CPU + 科学对照价值,无生产价值。

## 26. Final Recommendation

**冻结 `sharded_v2` payload = Option A(hybrid 仅一处,由语义而非压缩驱动):**

```text
continuous fields(除降水):   plain float16(<f2, little-endian)+ Zstd L5
precipitation_amount_3h:      float32(原生 f32,同 v1)—— 阈值语义例外
categorical det/mem flags:    uint8
GEFS mean-shard flags:        float32(成员均值概率,维持现状)
compute / cache 读取后物化:   float32(不变)
```

- 预期收益:全周期 **−22.0%**(f16-everything 为 −22.4%;降水保持 f32 放弃 ≈0.4pp 买断 exactly-0.1 风险),active ≈ −22%;Range GET −18~19%;chunk-cache payload −50%;RAM 记账同 REPORT.md §10。
- **降水的三选一**:默认按上表 (iii) f32——零新机器、fallback 前驱路径与 REPORT.md §6 完全一致、对今天产品零位移。若产品方确认"exactly-0.100mm 判干更符合政策意图"(基线判湿本身是 float 展开噪声),可改选 (i) plain f16(多 0.4pp 节省,需 sign-off);(ii) round(0.01mm)+f16 仅当未来降水压缩被证明值得一个单变量量子。
- **明确否决**:quantized f32 全家族(C/E/G:压缩差 12-19pp,RAM 2×)、round1/0.05mm 类网格(阈值病理)、fixed-point(全周期 +0.1%)。
- **不做 hybrid per-variable 压缩策略**:全周期加权下没有任何量化变体对 plain f16 有 ≥4pp 的实质优势,而 D2 的 +3.6% 需要永久策略——不满足"额外收益足够大"的 hybrid 门槛。

## 27. 决策原则复核(对照本轮实测)

- "plain f16 接近 Pareto optimal → 推荐 plain f16":**命中**(D2 全周期 +3.6% 相对,非假设中的 24% vs 22%,而是 25.2% vs 22.4%,且伴随真实副作用)。
- "quantized f16 显著更优(30%+)→ 考虑":**未命中**(实测 25.2%)。
- "quantized f32 压缩很强也要看 RAM":**命中且加强**(G11:同等精度压缩也只有 8.3%,RAM 2×,双重出局)。
- "fixed-point 仅多 3-5% → 否决":**命中且加强**(实测 +0.1%)。

---

### 附:可复现性与对上一轮报告的修正
- 脚本:`%TEMP%\f16bench\{bench6_quantization,bench7_cpu_cycle}.py`;结果:`results\{bench6_quantization,bench7_cpu_cycle}.json`;数据:复用上一轮 85 字段(未重新下载)。
- **对 REPORT.md 的修正**:其 §5"GEFS 降水 trace 翻转 = 0"与 §6 相关结论在 float32 比较域成立;生产代码在 float64 域比较(`precipitation.py:184`),exactly-f32(0.1) 格点(GEFS 场 12.813%)在 plain f16 下湿→干。已在 REPORT.md 末尾追加 addendum 指向本报告 §14。
- 未采样项:cloud_ceiling(周期估算用 cloud 类比,est);GEFS mean-shard flags(不在本轮范围)。
