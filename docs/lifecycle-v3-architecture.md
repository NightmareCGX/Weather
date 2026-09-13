# 数据生命周期架构说明（Lifecycle V3 · 收敛版）

> 状态：权威设计文档。本版整合 Lifecycle V3 锁定契约与 2026-09 代码审计后的八项收敛修正。
> 上一版中 "Whole-cycle finalizer 作为 GC 层" 的表述已被废除，统一为 canonical-necessity
> 单一权威模型。
> 代码映射：`packages/domain`（纯逻辑）、`services/api`（serving）、`services/ingestion`
> （realtime 采集与 GC）。

---

## 0. 最终收敛后的核心模型

```text
Realtime ingestion
  → variable / lead / member 独立 commit 与 publish

Serving
  → variable / valid_time canonical source resolution

GC
  → variable / valid_time 独立判定必要性
  → individual physical units 独立回收

Cycle lifecycle
  → 仅 derived bookkeeping

Batch ingestion finalize
  → batch-only operational concept
  → 不参与 realtime serving / GC authority
```

**GC 最终 invariant（锁死）：**

```text
There is no cycle-level GC authority.

A physical reclamation unit may be deleted only when:

1. canonical serving no longer selects it for any protected valid_time;
2. it is not held by interval fallback;
3. it is not held by precipitation companion source coupling;
4. it is not held by wind U/V atomicity;
5. it is not held by predecessor/recovery dependencies;
6. ensemble/member integrity does not require it: while a (cycle, lead) remains
   a canonical ensemble source, every committed member of that unit is retained —
   the 85% coverage threshold is a candidate-eligibility bound, never a
   reclamation target;
7. its departure from the protected set is durable: for a valid_time still inside
   the active serving window, the replacement serving state has been durably
   published; for a valid_time that has exited the active serving window, the
   exit is established by the monotonically non-decreasing serving_start
   boundary and requires no replacement publication;
8. deletion-time counterfactual revalidation still proves it unnecessary.

Cycle-level deleted_at is only a derived bookkeeping fact after all
constituent units have independently become terminal.
```

---

## 1. 核心原则

1. **用户面对 valid_time**。前端与 serving 语义围绕 `model + variable + valid_time`；
   `cycle_time / lead_time` 保留在后端，作为 provenance、cache identity 与
   ingestion / lifecycle identity，不再是用户交互模型。
2. **三个概念正交，不得混用**：
   - **A. UI visibility / expiration**：valid_time 何时从下拉框消失；
   - **B. Serving eligibility**：API 对哪些 valid_time 具有 serveability guarantee；
   - **C. Physical deletion**：物理 unit 何时可删。
3. **Canonical source resolution 是 GC 的唯一删除权威**。删除判定始终是：
   > 某 physical reclamation unit 是否仍被任何受保护的 variable/valid_time canonical
   > serving 或 dependency hold 所需要。
   `T−C`、`T−2C` 等 cutoff **不是**与 canonical resolver 平行的第二套 deletion
   authority，而是当前 GFS/GEFS 配置下的**派生推论 / planner fast-path 优化**。
4. **GC 完全解耦到 (variable, valid_time) 粒度**。各变量之间、各 valid_time 之间的
   过期回收互不依赖；不存在 cycle 级 GC authority。
5. **Realtime 的 serving/failure 单元是 variable-lead（及 GEFS member）**，不是
   whole-run。`model_runs` 状态仅是 operational/catalog 信息，不是 serving 或 GC 的
   一刀切 authority。
6. **Lifecycle identity 是 `(model_id, cycle_time)`**。GFS/GEFS 独立推进。

---

## 2. 独立的时间元数据（互不绑定的四个概念）

以下数值当前恰好匹配，但架构上是**不同概念**，必须保持独立注册：

| 元数据 | 当前值 | 语义 |
|---|---|---|
| model/product max lead | GFS=240h、GEFS=240h | 产品的 forecast horizon |
| model cycle cadence | GFS=6h、GEFS=6h | 相邻 cycle 的间隔 |
| variable interval width | precipitation=3h、cloud=3h | 区间量的统计宽度 |
| variable reset period | precipitation=6h、cloud=6h | 上游累积量的重置周期 |

因此架构语义上：

```text
L % 6 == 0            （错误：绑定了某个具体数字）
L % variable.reset_period_hours == 0   （正确）
```

未来引入不同 horizon 或 reset 语义的产品时，不会把 model cadence 与 variable
accumulation/reset 语义错误耦合。代码映射：`domain/horizon.py`（max lead）、
`domain/cadence.py`（cycle cadence）、`domain/temporal.py`（interval 语义）、
`domain/reclamation.py` 的前驱判定应参数化为 `reset_period_hours`。

valid_time 网格与 anchor：`serving_start = latest model valid_time <= now`
（floor 到 3h 网格；07:00Z → 06Z，09:00Z → 09Z）。anchor 推进 → 前端强制刷新 →
旧 valid_time 退出 active serving window。

---

## 3. 变量语义分类（静态契约，绝非数值判断）

| 类别 | 变量 | lead 0 语义 |
|---|---|---|
| Instantaneous | temperature_2m、wind_10m（u/v 合成）、relative_humidity_2m、visibility、snow_depth、cloud_ceiling … | 有意义，T+0 即 canonical |
| Interval | precipitation_amount_3h、cloud_cover_3h | store 中 NaN，必须 fallback 到正 lead |
| Precip companions | crain、csnow、cfrzr、cicep | 非独立 resolver 变量；严格跟随 precipitation_amount_3h 实际 source |
| Wind 分量对 | wind_u_10m + wind_v_10m | **原子对**：U、V 都 commit ready 才进入 serve；GC 成对进退 |

分类只来自变量名静态集合（`domain/temporal.py`），**绝不允许** `if value == 0` 或把
NaN 当 serving contract——0 mm 降水是真实预报。

**Reset lead 重建（按 W、R 一般化）**：设 `W = variable.interval_width_hours`、
`R = variable.reset_period_hours`（要求 `W < R`；`W == R` 时上游在 reset lead 已给出
W 宽度，无需重建）。仅在 reset lead（`L % R == 0`）需要重建：

```text
precipitation accumulation:
    A_interval(L) = A_reset_accum(L) − A_reset_accum(L − W)      前驱 = L − W

cloud running average:
    X_W(L) = [ R·C(L) − (R−W)·C(L−W) ] / W
```

当前配置 `W=3, R=6` 分别退化为 `f006 − f003` 与 `2·C₆ₕ − C₃ₕ(t−3)`（cloud 带
物理范围 guardrail）——现有公式只是特例，**禁止**把 `W = R/2` 写进通用逻辑。
差分在 **ingestion 解码层**完成，serving 读到的已是区间值；GC 前驱 hold 见 §7.1
（reset lead L hold 同 run L−W）。

---

## 4. Serving 模型（source selection）

对每个 `(model, variable, valid_time)`：

- **R1** 所有能产生该 valid_time 的 `(cycle, lead)` 组合都是候选。
- **R2** **newest serveable wins**：最新的安全 committed 且 serveable 的候选胜出。
  新 partial cycle 只在它实际 committed 的 valid_time 上覆盖旧 cycle。
- **R3 interval lead0 fallback**：仅当 winner `lead == 0` 时触发；目标为同一
  valid_time 的 **next-newest serveable positive-lead** 候选；正 lead 永不 fallback。
- **R4 companion 同源**：crain/csnow/cfrzr/cicep 与 precipitation_amount_3h 完全同
  (cycle, lead, store)；当前 cycle lead0 的 categorical flags 是 ingestion 合成表示，
  混源会 phase 错判。
- **R5** cloud_cover_3h 独立 fallback，与 precipitation 不耦合。
- **R6 wind 原子对**：wind_10m 要求 u、v 同候选；U、V 都 commit ready 才可 serving。
- **R7 coverage threshold**：GEFS 每 lead ≥85% member（整数安全判定）；fallback
  候选同样受限——fallback 是 "next-newest serveable positive-lead candidate"，
  不是 "previous cycle"。
  **85% 是 candidate eligibility threshold，不是 GC retention target**：它只回答
  "这个候选够不够资格 serving"，不回答"最少需要保留多少 members"。只要某
  (cycle, lead) 仍是 canonical ensemble source，其**已 committed 的成员全部
  retained**——绝不能因 26/30 已满足 85% 而回收其余 4 个已 commit 的 member。
- **R8 graceful null**：无 fallback 时 instantaneous 照常出值、interval 置 null，
  整个请求仍成功。
- **R9 mixed-source 是正确形态**：/v1/points 一个 entry 内不同变量可来自不同
  source；valid_time 是统一时间轴。
- **R10 availability 真实语义**：interval 变量只有存在有效 positive-lead source 才
  available；instantaneous 不受此约束。
- **R11 cache 绑定真实 source**：单变量端点 key = 实际 (cycle, lead) + valid_time +
  该 source store 的 serving generation。/v1/points 的 provenance digest 对**每个
  实际使用的 source** 生成条目
  `(variable, valid_time, run_id, source cycle, source lead, 该 source store 的
  manifest serving generation)`（无 fallback source 时 NULL 哨兵），排序后哈希。
  generation 必须按各 source **自己的 store** 解析——不能只取 primary/anchor 的
  generation，否则 fallback source 所在 store 重新 publish（G06 17→18）而 primary
  未变时，缓存会命中旧 precipitation。companions 作为独立条目，但因 I10 与
  precipitation 条目同值（更保守：耦合被破坏时 digest 如实变化）。
- **R12 provenance 反映真实来源**：单变量端点报实际 `source_cycle`/`lead`；
  /v1/points entry 级保留 primary/anchor 以维持 API 兼容。

---

## 5. Serving retention 边界与 `protected_valid_times` primitive

```text
UI visibility、serving eligibility 和 physical deletion 是独立状态。

API 只保证 active serving window 内 valid_time 的 serveability；
valid_time 退出 active window 后不再具有 serving retention guarantee，
但物理数据也不要求同步删除。
```

**`protected_valid_times` 是共享 domain primitive（active serving window policy 的
唯一权威实现）**：

```text
protected_valid_times(model, now) =
    { valid_time : valid_time >= serving_start_valid_time(now) }
    ∩ 该 model 的 canonical horizon 网格
```

- 唯一实现位于 `packages/domain`（纯函数；现有 `serving_start_valid_time` 升级为
  其载体）。
- 三个消费者**引用同一实现，禁止各自计算边界**：
  1. 前端 availability——通过 API 暴露的 serving window 元数据消费，TS 侧不重复
     实现边界数学；
  2. API serving eligibility——resolver 的窗口左边界判定；
  3. GC canonical protection——planner 的 `start_valid_time` 参数与 worker
     deletion-time 复验。
- invariant 1/7 中的 "protected valid_time" 即由此 primitive 定义。UI/API/GC 三套
  boundary drift（"UI 认为 03Z expired、API 认为 active、GC 认为 protected"）在
  结构上不可能发生。

产品行为链：anchor 推进 → 前端强制刷新 → 旧 valid_time 退出 active serving window →
用户不再能通过正常产品交互显式请求旧 valid_time。GC **不需要**为"用户可能继续查询
已退出窗口的 valid_time"保留数据；同时物理删除也不要求与窗口退出同步。

UI 侧机制：3h grace 窗口过滤 + 60s 心跳 + 与 cadence 边界对齐的定时刷新。

---

## 6. Realtime ingestion：variable-lead-member commit 语义

### 6.1 commit / publish 单元

serving 与 failure 的单元是 **variable-lead（及 GEFS member）**，不是 whole-run：

```text
failed / uncommitted variable-lead unit
  → 不参与 canonical serving，也不产生 serving hold；

同一 cycle 中其它已 committed / published 的 variable-lead units
  → 不受影响。
```

写入时序（每次 wave）：

1. **reserve_run**（物理写之前）：upsert lifecycle 行 + `model_runs('processing')`。
   catalog 身份先行。
2. **region commit**：EXCLUSIVE gate 下写 variable shard，region COMPLETE marker
   最后写（store 侧最后一步）。
3. **lead settlement / promotion**：lead 全部成员 settle 后 `publish_settled_lead`：
   按 marker 证据写 EnsembleMember / EnsembleMemberProduct / Product 行并 bump
   manifest serving generation。此步不标记 run ready。
4. **batch finalize（batch-only）**：`model_runs` ready/partial 状态由 batch ingestion
   finalize 推导，是 **operational/catalog 概念**，不是 realtime serving 或 GC 的
   authority。

### 6.2 wave 调度模型

单 leader、wave 串行派发（GFS → GEFS，同模型永不并发）；新 wave 只在上一个 wave
完成 finalizer 后开始；优雅停机走 non-abandoning drain；scheduler 无自有状态，
每次 poll 从上游 snapshot + durable catalog 重建 pending，崩溃后下一轮
reconciliation 只补缺失。**当前设计下 wave 不会被永久弃跑**；实测孤儿 store 是
历史遗留（catalog 接线前路径 / 库重建），防复发靠 §9 对账。

---

## 7. 物理 GC：canonical necessity 单一权威

### 7.1 删除权威与派生 fast-path

**删除判定始终是**（§0 invariant 的 1–8 条），由与 serving 完全同一套 canonical 纯引擎
（`select_canonical_sources_bulk`）执行。

在当前 GFS/GEFS 配置下，该判定的**派生结果**可以写成 cutoff 形式，作为 planner 的
fast-path / 审计口径（T = latest ready 且已开始 serving 的 cycle，C = cycle cadence）：

- instantaneous 变量 shard：`cycle_time ≤ T − C` 后即不再被 canonical 选中；
- interval 变量 shard：T−C 仍作为 lead0 fallback 在 serving，故 `cycle_time ≤ T − 2C`
  后才不再被选中；
- wind：u/v 原子对，T 的 wind 开始 serving ⇒ T−C 的 u/v 均不再被选中。原子性是
  **semantic atomicity**，不是物理事务（定义见 §7.2）。
- predecessor：reset lead（`L % variable.reset_period_hours == 0`）的 shard 依赖
  同 run **L − variable.interval_width_hours**（不是 L − R/2）；相关前驱未 commit 前
  （恢复场景）被 dependency hold。

这些 cutoff 永远是**推论**，不是判定来源；配置变化时以 canonical 解析为准。

### 7.2 执行管线（只有这一条）

```text
Reclamation Planner
  → （canonical 解析 + dependency holds）→ reclamation_queue
Reclamation Worker
  → 租约领取 → SHARED store gate → deletion-time counterfactual revalidation
  → DeleteObject（unit 级）→ unit terminal
（循环，直到所有 constituent units terminal）
```

- **Planner**：对每个 (model, variable, valid_time) canonical 解析；非 anchor、非
  variable fallback source、非 wind 连带、非 predecessor、member integrity 不需要的
  committed unit → 入 `reclamation_queue`。
- **Worker**：租约领取批次 → SHARED store gate → 重读 lifecycle（若 cycle 已
  tombstone 则按 bookkeeping 语义处理）→ counterfactual revalidation（本批视为
  物理在场、其余 deleting/deleted 视为 fenced，重算必要性；仍必要者回退 queued）→
  DeleteObject → 标记 deleted。**每个 physical unit 独立通过全部 8 条后才删除。**

  **Wind pair 的 semantic atomicity**（S3 DeleteObject 无跨对象事务，原子性分层
  定义）：
  1. **Eligibility atomic**：u 与 v 必须同时判定为 unnecessary 后，才允许任何一个
     进入 `deleting`——任一分量仍必要，则 pair 整体不入删除批；
  2. **Serving atomic**：pair 一旦被 fenced / reclaiming，resolver 不得再选择该
     pair（现有 `filter_candidates_by_physical_fence` 已提供此语义：fenced 的 u 使
     候选失去 wind_u → wind_10m 的"同候选 u+v"要求使整个 pair 不可选）；
  3. **Physical deletion**：u 与 v 可以依次 DeleteObject，不要求同时；
  4. **Crash between deletes**：允许短暂 one-deleted / one-present 状态——pair 已
     整体 fenced（serving 不受影响），recovery 对仍在 deleting 租约内的剩余 member
     继续删除直至 terminal。
- **不存在 whole-cycle GC finalizer**：不存在"整 cycle GC → 决定是否删除整个 cycle"
  的阶段，也不存在基于 `cycle horizon expired` 直接整 prefix 删除的独立 authority。
  当前实现中 horizon-based 的 `run_finalizer_pass` 删除路径必须退役（见 §11）。

### 7.3 Marker / manifest cleanup：同样 dependency-driven

S3 prefix 不是实体目录，**没有** "rm 整个 cycle prefix" 的最终阶段。每种 metadata
object 按自身依赖独立回收：

```text
variable shard
  → 对应 canonical/dependency hold 消失后删除（worker 现行路径）

region COMPLETE marker
  → 它所证明的全部 physical units terminal 后删除（现行 region marker 清理即是此模型）

manifest / cycle metadata object（.zattrs/.zgroup/.zmetadata 等）
  → 所有仍依赖它的 active units terminal 后删除
```

GC 从头到尾维持同一个 dependency-driven 模型；cycle store 前缀的"清空"是所有
unit 独立删除后的**结果**，不是任何一步的**动作**。

### 7.4 Failed / uncommitted unit 与 partial 保护

- failed / uncommitted variable-lead unit 不参与 serving、不产生 hold（§6.1）；
- 比"当前 serving 的 cycle"**更古早**的孤儿/落后 unit：canonical 解析自然不再选中，
  可回收；
- 最新的、后面还没有 ready cycle 的 partial cycle 中已 committed 的 units：处于
  serving 窗口内，被 canonical hold 保护，**不得仅凭时间删除**；
- 无 arbitrary TTL（明确拒绝 48h 超时类策略）。

### 7.5 删除硬前置：两类 GC 触发，"持久性"的证据不同

invariant 第 7 条按 unit 离开 protected set 的方式分两类：

**A. Replacement GC（窗口内被接管）**：valid_time 仍在 active window 内，由更新的
cycle unit 接管。硬前置 = 接替 source 的 manifest serving generation 已 bump——
这是 serving 切换的持久证据，可被 planner/worker 验证。前端强制刷新由 API 侧
generation 变化驱动，GC 不直接感知 UI。

**B. Window-exit GC（窗口退出）**：valid_time 本身退出 active window
（`VT < serving_start`）。**不存在 replacement**——serving_start 是 UTC 的确定性
函数、单调非降，退出即自证，不需要任何 publication。由于下一 cycle 能产生的最小
valid_time 是 `cycle_time + cadence`，每个 cycle 的 **lead-0 / lead-3 unit（一般地：
VT 低于下一 cycle 首个 VT 的所有 unit）以及 interval 变量的 anchor-region unit**
永远不会被替换，只走这一类。若强制要求 replacement generation，这些 unit 将永远
无法回收。

安全性依据：① serving_start 单调非降，退出的 valid_time 不可能重新进入窗口；
② 条件 8 的 deletion-time counterfactual revalidation 在删除瞬间重算 protected
set，时钟回拨等瞬态由最后一道复验兜底；③ 可选 belt-and-braces margin（仅回收
`VT < serving_start − margin`），非必需。

---

## 8. Cycle lifecycle：derived bookkeeping only

`forecast_cycle_lifecycle.deleted_at` 继续永久保留，但语义明确为：

```text
该 cycle 的所有 physical reclamation units 都已独立进入 terminal state
```

而不是"deleted_at / 某 finalizer 授权删除整个 cycle"。方向只能是：

```text
per-variable/per-valid-time GC
  → 所有 constituent units terminal
  → derive cycle-level deleted_at（启动 14-day metadata retention clock）
```

不能反过来。当前实现中承担该角色的组件应改名为
**Lifecycle bookkeeping / metadata reconciliation**，它只允许：

- 观察所有 units 是否 terminal；
- 写 `deleted_at`；
- 启动 14-day metadata retention clock（sweeper 清明细 catalog：
  model_runs / forecast_products / ensemble_members / ensemble_member_products /
  reclamation_queue；tombstone 永久保留）。

它**不允许**：DeleteObject、删除整个 prefix、决定某 shard 是否回收、绕过 canonical
planner。

---

## 9. Store ↔ catalog reconciler：单调 recoverability frontier

系统具有单调性：**旧 cycle 不会重新被 realtime scheduler 激活**。因此对账不需要
anti-resurrection 子系统，采用简单规则：

```text
发现 store exists + catalog missing
        ↓
判断 cycle 是否仍位于 recoverable / active frontier 内

已越过 recoverability frontier
        ↓
永不恢复 catalog
        ↓
按 orphan cleanup 处理（units 逐个走 canonical necessity 判定后回收；
实际上此时它们全部不必要，可快速 terminal）

仍处于合法 recoverable region
        ↓
检查 COMPLETE marker / durable evidence
        ↓
允许恢复 catalog
```

永久 lifecycle tombstone 可作为额外 sanity guard，但不围绕 resurrection 单独建立
新机制。

---

## 10. 不变式清单

| # | 不变式 |
|---|---|
| I1 | 用户语义仅 `model + variable + valid_time`；cycle/lead 仅存在于后端 |
| I2 | anchor = latest model valid_time ≤ now；边界推进触发前端立即刷新 |
| I3 | UI visibility、serving eligibility、physical deletion 三者独立；API 只保证 active window 内的 serveability |
| I4 | lifecycle 身份为 (model_id, cycle_time)；模型间零耦合 |
| I5 | GC 判定粒度 (variable, valid_time)；**canonical necessity 是唯一删除权威** |
| I6 | cutoff（T−C / T−2C）是派生 fast-path，不是 deletion authority |
| I7 | **No cycle-level GC authority**；unit 删除须通过 invariant 1–8 全部条件 |
| I8 | wind u/v **semantic atomicity**：eligibility atomic（u/v 同时判 unnecessary 后才允许任一进入 deleting）＋ serving atomic（pair 一旦 fenced/reclaiming，resolver 不得再选择该 pair）；物理 DeleteObject 可顺序执行，crash 产生的短暂 one-deleted/one-present 由整体 fence 覆盖、recovery 删除剩余 |
| I9 | reset lead 依赖同 run 前驱 `L − W`（W=interval width，非 R/2）；前驱未消费前被 dependency hold；适用性由 `L % R == 0` 判定 |
| I10 | companions 与 precipitation_amount_3h 永远同源 |
| I11 | fallback 只由语义类别 + lead==0 触发；绝不依据数值/NaN |
| I12 | fallback 候选必须通过 coverage threshold |
| I13 | realtime serving/failure 单元是 variable-lead-member；whole-run 状态仅是 operational/catalog 信息 |
| I14 | unit 离开 protected set 必须持久：窗口内被替换 → replacement generation 已 bump；窗口退出 → 由单调 serving_start 自证，无需 publication |
| I15 | 最新且无后继的 committed partial 不得仅凭时间删除；古早孤儿可回收 |
| I16 | `deleted_at` 是 derived bookkeeping；tombstone 永久保留；明细元数据 14 天 |
| I17 | cache identity 与 provenance 绑定实际 source 与 serving generation |
| I18 | marker / manifest cleanup 与 shard cleanup 同为 dependency-driven；不存在整 prefix 删除阶段 |
| I19 | max lead / cycle cadence / interval width / reset period 是独立元数据；重建公式只以 (W, R) 表达，禁止 `W = R/2` 隐式绑定 |
| I20 | `protected_valid_times(model, now)` 是 active serving window policy 的唯一权威 primitive；前端/API/GC 引用同一实现，禁止各自计算边界 |

---

## 11. 当前实现与目标态差距

1. **退役 horizon-based whole-cycle 删除路径**：现行 `run_finalizer_pass` 基于
   `cycle_time+240h < serving_start` 执行整 prefix DeleteObject——这正是被废除的
   "cycle-level GC authority"。改造方向：删除动作全部收敛到 planner+worker；
   finalizer 降级为 §8 的 Lifecycle bookkeeping（只观察 terminality、写 `deleted_at`、
   启动 14 天时钟，不做任何物理删除）。
2. **V2 reconciler 残留下线**：其 `T − cadence` cutoff + 立即删明细元数据的行为与
   I6/I16 冲突。
3. **Planner 快路径落地**：以 canonical 解析为权威，cutoff 作为批处理剪枝/审计口径
   实装（避免全历史扫描）；配合 batch_size 分批入队。
4. **worker 复验的 wind 连带成对化**：按 R-WIND/I8 成对评估 u/v，显式化原子性。
5. **/v1/points fallback 实现**：与 canonical 引擎平行的第二套实现；fallback store
   缺 precipitation 变量时 404 击穿整个请求（违反 R8）——收敛到同一引擎或在选择时
   校验变量集并继续尝试下一候选。
6. **ensemble_data valid_time 路径读后 fence 复核静默失效**（`run` 未定义 + `db`
   已关闭 + `except: pass`）。
7. **store↔catalog 对账任务（§9）待建**：recoverability frontier 规则 + 历史孤儿
   （gefs 2026-09-12/00、06 等 ≈57GB）清理。
8. **时间元数据拆分（I19）与重建公式一般化**：`domain/reclamation.py` 的
   `is_predecessor_dependent_lead`（硬编码 `% 6`）与 `get_predecessor_lead`
   （硬编码 `L − 3`）按 `(W, R)` 参数化为 `L % R == 0` / 前驱 `L − W`；
   `reconstruct_cloud_cover_3h`（硬编码 `2·C6 − C3`）改为
   `[R·C(L) − (R−W)·C(L−W)] / W`。
9. **availability legacy `initial_times` 视图**仍将 interval lead0 列为 servable
   （权威 `valid_times` 正确）。
10. **测试污染生产 catalog**：pytest 写入真实 DB（`pytest-of-Ezrai` 临时路径出现在
    `model_runs`）——测试需隔离实例。
11. **提取 `protected_valid_times` primitive（I20）**：将 `serving_start_valid_time`
    升级为 per-model 的 protected set 权威实现；API 在 availability/响应元数据中
    暴露 serving window，前端消费同一策略结果，删除各处对边界的重复计算。
