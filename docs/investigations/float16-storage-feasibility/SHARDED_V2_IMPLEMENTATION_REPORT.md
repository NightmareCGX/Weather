# Sharded V2 Implementation Report

实施依据:`SHARDED_V2_IMPLEMENTATION_PLAN.md`(authoritative)+ `REPORT.md` + `QUANTIZATION_BENCHMARK.md`。本轮完成 Phase 0–8 的全部代码、测试、shadow tooling、配置支持与文档;**未做任何生产 rollout,未 commit/push**。

---

## 1. Summary

**SHARDED V2 IMPLEMENTATION COMPLETE**(§四十二 25 项验收标准全部满足,见 §20 逐项核对)。冻结存储合同已全链路落地:9 个连续量 `<f2`、`precipitation_amount_3h`/`cloud_ceiling` 两个 f32 语义例外、det/member flags `u1`、mean flags `<f4`、显式小端、Zstd L5、index/trailer/对象键与 v1 完全一致、版本由 manifest `storage_format_version="sharded_v2"` 区分。生产灰度(GFS→GEFS mean→GEFS members)保留 GO/NO-GO,能力已就绪。

## 2. Files Changed

**修改(13 个)**:
- `packages/domain/src/domain/storage_dtype.py` **(新)** — authoritative dtype resolver:冻结矩阵、`ProductRole` 复用 `TARGET_KIND_DET/MEAN/MEM`、fail-closed(未知变量/未知格式 raise)、`is_sharded_payload_format` 谓词。
- `packages/domain/tests/test_storage_dtype.py` **(新)** — 14 个 resolver 测试(矩阵、角色依赖、fail-closed、小端、15 变量完备性、集合不相交)。
- `services/ingestion/src/ingestion/core/base.py` — 新异常 `F16RangeViolationError`、`CategoricalDomainViolationError`、`CycleFormatConflictError`(全部带合同文档)。
- `services/ingestion/src/ingestion/core/config.py` — `STORAGE_FORMAT_VERSION` 文档扩展(v2 语义、per-cycle 冻结、rollback),默认仍 `sharded_v1`。
- `services/ingestion/src/ingestion/core/pipeline.py` — **Phase 0 flags 短路修复**:`_normalize_canonical_units` 在 unit token 已为 "flag" 时仍强制 `np.uint8`(det/member categorical 稳定 uint8,不再依赖 writer 猜测)。
- `services/ingestion/src/ingestion/core/zarr_writer.py` — `encode_region_sharded_v2`(per-variable native dtype 编码器)+ `encode_region_sharded` dispatch + writer guards(`_validate_f16_block`:NaN 放行计数、±Inf/overflow 在 cast **前**拒绝;`_validate_categorical_block`:det/mem flags ⊆{0,1}、mean flags ⊆[0,1])+ `assert_cycle_format_allowed` 进程内格式冻结 + `commit_region` 双 dispatch 块接 v2(guard 违例**不**静默回退 legacy)+ `prepare_run_store` v2 分支(`.zarray` dtype=resolver 结果,修复 v1 的元数据/字节不一致;v1 行为冻结不变)+ `read_slice`/`_populate_sharded_data` dtype-aware(manifest 权威;manifest 缺失时按解码字节长度从 resolver 候选推断——关闭"未发布 v2 store 被当 f32 读"的隐藏路径)+ `_assert_chunk_byte_length` 字节长度最后一道防线。
- `services/ingestion/src/ingestion/core/coordinator.py` — cycle format freeze 持久层:`_assert_cycle_format_freeze` 在 finalize 与 per-lead publish 两个 manifest 站点对比已提交 manifest 的版本,不一致 raise(重启安全的第二层防线)。
- `services/ingestion/src/ingestion/core/inventory.py` — `region_expected_object_keys` 接受 sharded_v2(v1/v2 物理键布局相同)。
- `services/ingestion/src/ingestion/core/shadow.py` **(新)** — shadow validation core:重编码、存储/数值/serving 三方比较、namespace 守卫清理。
- `services/api/src/api/core/zarr_v2.py` **(新)** — `ShardedV2Reader`(继承冻结的 `ShardedV1Reader`,仅覆写 chunk 解码;**native dtype chunk cache**:f16 缓存为 f16、u8 缓存为 u8;字节长度守卫)。
- `services/api/src/api/core/zarr.py` — `get_sharded_reader` 按 store manifest 分派 V1/V2 reader(格式 per-store 不可变,LRU key 仍为 path)。
- `services/api/src/api/services/{point_forecast,tiles,ensemble_data,vector_field}.py` — 6 个 serving 分支点 `== "sharded_v1"` → `is_sharded_payload_format(...)`(覆盖 point/hourly/tiles/ensemble/wind/phase 全部入口)。
- `services/ingestion/tests/test_pipeline.py` — flags 短路回归测试。
- `.env.example` — `STORAGE_FORMAT_VERSION` rollout 条目(灰度顺序、冻结、rollback 说明)。

**新增测试(3 个文件)**:`test_sharded_v2_writer.py`(25 用例)、`test_sharded_v2_reader.py`(13 用例)、`test_shadow_validation.py`(7 用例)。
**新增工具**:`scripts/shadow_v2.py`(`validate`/`cleanup` 子命令)。

## 3. Final Storage Contract

与 PLAN §4/§5 完全一致(逐项核对 ✓):dtype matrix、显式小端 `<f2`/`<f4`/`u1`、Zstd L5、100×100 几何与边缘 chunk 特例、SHAR magic/index/trailer/对象键不变、`storage_format_version="sharded_v2"`、`.zarray`==payload dtype(仅 v2)、resolver 单一事实来源(全仓无第二份 FLOAT16_VARS 副本)。`v2_unsharded` legacy 字符串未被纳入 v2 路径(命名警示已写入两处 docstring)。

## 4–10. Phase 0–8 Results

| Phase | 内容 | 结果 |
|---|---|---|
| 0 | flags 短路修复、resolver、异常类 | 完成;`test_pipeline` 3/3、`test_storage_dtype` 14/14 |
| 1 | format contract、双层格式冻结、`.zarray` dtype | 完成;freeze 双测试;`.zarray`==payload 断言(v1 行为冻结对照) |
| 2 | `ShardedV2Reader`、工厂分派、6 serving 分支 | 完成;13/13;native cache 断言(缓存数组 dtype=f16) |
| 3 | v2 writer、guards、hidden readback、inventory | 完成;25/25;全仓 frombuffer/4B 假设复核(`wind.py` i16 与 `png.py` RGBA 与存储无关) |
| 4 | shadow tooling + **本地可执行验证** | 完成;CLI 实测 exit=0(见 §16) |
| 5–7 | GFS/GEFS mean/GEFS members rollout 能力 | env/config/冻结/rollback/reader 兼容全部就绪;**未擅自生产切换**(GO/NO-GO 3/4/5 保留) |
| 8 | v1 natural retirement | v1 字节合同冻结、v1 精确测试全绿、mixed-format serving 测试通过;lifecycle 零改动(无 lifecycle 文件出现在 diff 中) |

## 11. V1 Compatibility Evidence

- v1 全部既有测试未放宽、未修改断言:`test_sharded_storage.py`、`test_sharded_reader.py`(33)、`test_serving_chunk_equivalence.py` 等 797+457 个用例全绿。
- v1 `.zarray` 历史行为冻结:新写 v1 store 的 flags 元数据仍为 seed dtype(f32),与历史一致(P2 修复只影响 normalized dataset 的 dtype 稳定性,不改 v1 字节/元数据)。
- `read_slice` 对 v1 store 返回 float32(冻结);`_populate_sharded_data` v1 路径 f32 frombuffer 不变。
- 全部 v1 dispatch 语义保留(guard/编码失败 → legacy 回退;仅 v2 的 guard 违例改为 fail-loudly,属 v2 新合同)。

## 12. V2 Numerical Validation

字节级:f16 chunk 恰 20,000B、f32 恰 40,000B、u8 恰 10,000B;LE 字节断言(`struct.pack("<e", 290.0)` 逐字节比对)。值级:v2 round-trip 后 f16 变量值 = `np.float16(源值)`;precip/ceiling f32 **逐位相同**。serving:f16 插值误差 ≤ f16 半 ulp;缓存 dtype 断言防 hidden astype。命令行 shadow:温度最大差 0.0076°C、其余变量 0.0。

## 13. Precipitation Semantic Regression

`test_precipitation_exactly_0_10mm_behavior_preserved`:v1 与 v2 双格式下 `np.float32(0.1)` 经存储→reader→`float()`→生产谓词(`<=0.10` in float64)分类完全一致(湿),且存储位模式逐位相等。永久保留,防止未来误把降水加入 f16 集。

## 14. Cloud Ceiling Semantic Regression

`test_cloud_ceiling_sentinel_boundaries_preserved`:19.990/19.991/19.992/19.993/20.000 km 五个边界值经 v2 存储(f32 例外)后分类与基线逐值一致;`>=19.99 → unlimited` 行为不变。永久保留。

## 15. Categorical Validation

det/member flags:u8、1 字节/值、值域 ⊆{0,1}(违例 raise `CategoricalDomainViolationError`);mean flags:`<f4`、4 字节/值、[0,1](违例 raise)。变量名相同但 role 不同 → resolver 强制分派, mean flags 不可能被 u8 化(测试覆盖)。

## 16. Shadow Validation Result

**已实际执行**(本地完整链路,非仅 tooling):`python scripts/shadow_v2.py validate --store <v1 cycle> --sample-points 40` → `SHADOW VALIDATION PASSED, exit=0`。
- 存储:v2 payload < v1(合成噪声场上 −49.5%;真实平滑场按基准轮实测为 −13~24%,合成结果偏乐观,已知偏差);
- 数值:precip/ceiling diff = 0.0、翻转 = 0;crain 精确;温度 max 0.0076°C;
- serving:40 采样点零阈值翻转,reader 工厂按 manifest 正确分派 V2;
- 工具守卫:cleanup 拒绝非 shadow 前缀、dry-run 默认、age guard 测试通过。
- 真实生产 shadow 命令:`python scripts/shadow_v2.py validate --store s3://weather-data/{model}/{date}/{HH}/cycle.zarr`(同 store 树内生成 `shadow-v2/`,不入 catalog)。

## 17. Performance Results

- **Writer 吞吐**(8 变量 region,Zstd L5):v1 248ms vs **v2 169ms(−32%,无回归)**——f16 将进入 Zstd 的字节量减半。
- **存储**:合成噪声场 −49.5%/region(真实气象场按基准轮 −13~24%,平滑场压缩比更高故 f16 增益小于噪声场,报告如实区分)。
- **API cache payload**:f16 变量 20,000B/chunk(−50%);测试断言缓存数组 native dtype。
- **Range GET**:f16 chunk 传输字节按基准轮 −18~24%(请求数不变)。

## 18. Tests

| 套件 | 命令 | 结果 |
|---|---|---|
| domain(含 100% 覆盖率门) | `uv run --package weather-platform-domain pytest packages/domain/tests/` | **548 passed,0 failed,覆盖率 100%** |
| ingestion 全量 | `uv run --package weather-platform-ingestion pytest services/ingestion/tests/` | **797 passed,29 skipped,0 failed**(skip=需 MinIO/PostgreSQL/Redis 服务) |
| api 全量 | `uv run --package weather-platform-api pytest services/api/tests/` | **457 passed,235 skipped,0 failed**(skip=需 PostgreSQL/Redis 服务容器) |
| 新增测试合计 | — | 59 个新用例全绿(14 resolver + 25 writer + 13 reader + 7 shadow) |
| ruff | `uv run ruff check packages/domain services/ingestion services/api` | **All checks passed** |
| mypy | domain/ingestion/api 变更文件 | 新增文件 0 错误;domain 29 个错误为实施前基线已存在(git stash 对照确认,数量未变) |
| frontend | — | **not run — reason: 本次未触及 frontend 代码** |
| CI-equivalent 容器/E2E | — | **not run — reason: 本机为 Windows,按 CLAUDE.md 由 ubuntu CI 执行** |

## 19. Remaining Production Actions(仅人工 GO/NO-GO)

1. **GO/NO-GO 3**:GFS det canonical 切换(env `STORAGE_FORMAT_VERSION=sharded_v2`,观察一个完整 lifecycle)。
2. **GO/NO-GO 4**:GEFS mean。
3. **GO/NO-GO 5**:GEFS members(最大容量收益所在)。
每步前可先跑 `scripts/shadow_v2.py validate --store <目标 cycle>`;回滚 = env 切回 `sharded_v1`(新 cycle 回 v1,已提交 v2 cycle 由 V2Reader 继续服务,零迁移)。

## 20. 验收标准核对(§四十二)

25 项逐条满足:v1 字节/读合同未回归 ✓;v2 writer/reader 完整 dtype matrix ✓;native cache ✓(测试断言);precip/ceiling f32 ✓(永久回归测试);det/member u8 ✓;mean flags f32 ✓;显式小端 ✓(字节断言);f16 guard ✓(cast 前拒绝);NaN 保留 ✓;Inf/overflow fail-loudly ✓;hidden readback dtype-aware ✓(含 manifest 缺失推断);cloud predecessor fallback ✓;mixed v1/v2 serving ✓;cycle format freeze ✓(双层+测试);rollback 实测 ✓;lifecycle 无语义改变 ✓;v1 测试未放宽 ✓;v2 数值测试完整 ✓;shadow 不污染 catalog ✓;相关测试套件全绿 ✓;lint/type 通过 ✓;无降低核心断言 ✓;本报告完整 ✓。

## 21. Risks / Follow-ups

- **Blocking**:无。
- **Non-blocking**:① manifest 缺失 + 首次 publish 前进程重启 + env 翻转的组合窗口(理论残留,operator 动作才能触发;manifest 校验在首次 publish 即拦截);② 未舍入 JSON 字面量在 f16 变量上的文本变化(与第一轮报告一致,展示层清理为独立 PR)。
- **Unrelated bug(未混入本 diff)**:`tiles.py:1263` legacy 成员路径 19990.0 单位错误;`docs` 四处过时描述(README GEFS 分辨率、ARCHITECTURE 缓存尺寸、finalizer 14 天、connector geavg 注释)——均按 PLAN §18 留给 release/docs phase。
