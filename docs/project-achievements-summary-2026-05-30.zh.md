# personal-semantic-memory-lab 成果总览

日期：2026-05-30

状态：工程 alpha / 研究原型。项目已经完成多条可执行闭环，但还不是面向普通用户的成品应用。

## 2026-05-31 更新：v0.41 统一入口验收

v0.41 usable-alpha workflow integration 已完成 mock/synthetic acceptance path 验收。

核心入口：

```powershell
python -m tools.workflow_runner ...
```

已验收通过：

- clean package probe；
- synthetic full build：S1 Build -> S1 Index -> S2 Build -> S2 Index；
- graph build：construction packets -> route -> extraction baseline -> consolidation -> NetworkX -> profile/community -> final quality gate -> visual bundle；
- query：lexical / embedding / graph 融合，自动发现 v0.3 graph dir；
- incremental human review：真实 WebUI 人工审核 session -> finalize -> S1 current / S2 current / graph current / invalidation / incremental visual refresh；
- full unittest：322 tests OK。

验收报告：

```text
docs/v0.41-release-acceptance-report.md
```

仍然 gated：

- bounded live provider smoke；
- public mirror 同步；
- durable memory write；
- graph truth；
- S3 / support checker authority。

## 一句话结论

当前项目已经从“个人语义记忆的概念实验”推进到一个有分层资产、路由、真实 provider、候选图构建、图查询、可视化审计、人工审核和增量 current view 发布的工程原型。

更准确地说，已经闭环的是：

```text
S0B/S1/S2 构建索引
  -> v0.21 资源校准路由
  -> v0.3 候选图构建与图感知查询
  -> v0.31 图可视化审计
  -> v0.4 人工审核驱动的增量 current view 维护
```

还没有闭环的是：durable memory 正式写入、S3 支撑检查器、图真值、生产级图数据库后端、最终实体合并策略、普通用户级产品体验。

## 核心架构

### S0B：原始材料组织层

S0B 负责把输入材料组织成可追踪的原始文本单元、section map 和 source range map。

当前设计原则：

- S0B 是底层证据组织层，不直接生成画像真值。
- v0.4 增量里，S0B 采用 add-oriented / append-only 思路。
- 删除、修改、废弃不优先物理改写旧行，而是通过 operation log、status 和 latest view 表达。

### S1：证据与记忆候选层

S1 负责把 S0B 材料变成 evidence、memory candidates、memory units 等证据绑定资产。

当前已经做到：

- route_s1 在 S1 Build 前强制执行。
- S1 LLM assist 可以进入 `processed_text`，但必须和 `original_text`、evidence refs、raw backpointers、warnings 一起移动。
- LLM 输出不是独立真值，不能替代原始文本。
- S1 Index 可以构建 BM25 / embedding 等可重建检索资产。

### S2：画像 / 用户模型层

S2 负责从 S1 资产生成 proposal-backed user model / portrait material。

当前已经做到：

- route_s2 在 S2 Build 前强制执行。
- S2 Build 消费 proposal outcomes，而不是直接吞 raw model output。
- S2 Index 可以构建用户模型检索资产。
- v0.4 中 S2 已经有 current view publish/merge：审核通过的新候选进入当前视图，旧受影响单元被标记为 historical/stale/excluded。

边界：S2 的 `reviewed_units` 在当前语境里仍是 proposal-backed / experimental material，不是 durable user-approved memory truth。

### Graph：候选图构建、算法、查询与审计层

v0.3 之后，图不再是 S2 的轻量投影玩具，而是单独的候选图工作流。

当前主链路：

```text
S1/S2 assets
  -> graph construction packets
  -> route_graph_extraction
  -> schema-guided provider extraction
  -> entity / relation / claim candidates
  -> consolidation / merge decisions
  -> NetworkX graph utility
  -> graph-aware retrieval
  -> S2 query integration
```

关键原则：

- 图构建是 LLM-primary、NLP-assisted、route-controlled、evidence-bound、candidate-first。
- GraphRAG / LightRAG 等成熟系统主要作为 pipeline 思路来源，不作为运行时依赖。
- regex / mock 只保留为 baseline 或测试，不是正式 fidelity path。
- 图指标、图可视化、社区检测都不是 correctness proof。

### v0.4：增量维护层

v0.4 已经把“新增/修改材料之后怎么更新上层资产”推进到可执行闭环。

当前闭环：

```text
new / changed evidence
  -> incremental candidates
  -> staged human review
  -> review session submit
  -> review decision applier
  -> S1 current view
  -> S2 current view
  -> graph current view
  -> dependency map / query invalidation
  -> affected visual review refresh
```

重要边界：

- S0B/S1/S2 历史仍然 append-only。
- current views 是派生投影，不是 canonical truth rewrite。
- human review gate 之后才进入 current publish。
- graph current 仍然是候选图当前视图，不是图真值。

## 版本成果

### v0.2：S0B/S1/S2 构建索引闭环

文档：`docs/v0.2-quality-upgrade-closure-report.md`

v0.2 关闭的是：

```text
S0B
  -> route_s1
  -> S1 Build
  -> S1 Index
  -> route_s2
  -> S2 proposal runner
  -> S2 Build
  -> S2 Index
```

核心成果：

- pre-build routing 成为 workflow policy。
- S1/S2 共用 router/proposal 基础设施，但 target task、prompt、schema、validation、write boundary 不同。
- S1 LLM assist 保持原文、处理文、证据和 warning 绑定。
- S2 消费 proposal outcomes。
- S0B、S1、S2、Index 边界清晰。

不代表：

- 最终语义质量成熟；
- 图语义成熟；
- S3 或支撑检查完成；
- 画像真值完成。

### v0.21：资源校准路由与真实 provider 构建索引闭环

文档：`docs/v0.21-build-index-closure-report.md`

v0.21 关闭的是：

```text
S0B/S1 intake
  -> route_s1 with v0.21 calibrated policy
  -> S1 provider lane
  -> S1 Build
  -> S1 Index
  -> route_s2 with v0.21 calibrated policy
  -> S2 provider lane
  -> S2 Build
  -> S2 Index
```

核心成果：

- 路由不再依赖临时玩具词表，而是转向外部资源、复杂度、salience、校准层。
- 默认 workflow 使用 `route_and_propose`。
- default provider lane 是真实 provider，不是 mock。
- S1/S2 都完成小批真实 provider smoke。
- S1 Index 和 S2 Index 均能在结果资产上重建。

默认状态记录：

```text
pre_build_route_mode = route_and_propose
route_policy = configs/routing/memory_proposal_router/heuristic_salience_v0.21.candidate.yaml
proposal_provider = openai
```

不代表：

- final route quality；
- final subject modeling quality；
- learned router runtime maturity；
- graph construction 完成。

### v0.3：候选图构建与图感知查询闭环

文档：

- `docs/v0.3-closure-report.md`
- `docs/v0.3-graphrag-core-code-learning-notes.md`
- `docs/v0.3-graph-tooling-and-algorithm-research.md`
- `docs/references/graph/v0.3-graph-toolchain-usage.md`

v0.3 关闭的是：

```text
v0.21 S1/S2 assets
  -> graph construction packets
  -> route_graph_extraction
  -> real provider schema-guided extraction
  -> candidate consolidation
  -> NetworkX utility
  -> graph-aware retrieval
  -> S2 query runner integration
  -> lightweight answer prompt view
```

核心成果：

- 学习 GraphRAG / LightRAG 后确认：先建图，再图算法；建图本身是核心层。
- 图构建主线改为 schema-guided LLM extraction，而不是继续扩 regex extractor。
- 每条 candidate 保留 evidence refs / source refs / raw backpointers / warnings。
- consolidation 明确保留 merge decisions，不做静默实体合并。
- NetworkX 用于第一阶段算法 utility proof。
- query 形成三路组合：字面、语义、图。
- S2 query runner 可输出轻量回答包，默认 prompt 不塞满证据链；why/audit 场景再展开证据路径。

v0.3 final gate 记录：

```text
status: pass_with_known_warnings
nodes: 171
edges: 233
claims: 171
evidence_links: 751
merge_decisions: 392
unresolved_edge_endpoint_count: 0
missing_evidence_rows: 0
generic_relation_ratio: 0.141631
```

不代表：

- 图是真值；
- 关系 ontology 已成熟；
- 实体合并已最终解决；
- 生产图数据库后端已选定；
- human-reviewed graph correctness 已完成。

### v0.31：图可视化审计闭环

文档：`docs/v0.31-graph-visualization-closure-report.md`

v0.31 关闭的是：

```text
v0.3 candidate graph assets
  -> bounded visual audit slices
  -> GraphML / JSON review bundle
  -> local HTML review page
  -> browser/external tool smoke
```

核心成果：

- 新增本地图可视化审计页面。
- 输出 GraphML / JSON，方便接 Gephi Lite、yEd Live 等成熟外部工具。
- 支持 ego network、community cluster、noisy generic edge、weak/isolated node、evidence path 等 bounded slices。
- HTML 页面支持点击边查看边信息和证据区域。
- 可视化定位为 audit support，不作为正确性证明。

不代表：

- 生产级图 UI；
- 图正确性证明；
- Neo4j / Kuzu / Memgraph 后端决策；
- durable graph truth。

### v0.4：人工审核驱动的增量维护闭环

文档：

- `docs/v0.4-incremental-maintenance-todo.md`
- `docs/v0.4-incremental-maintenance-workflow.md`
- `docs/v0.4-incremental-maintenance-external-patterns.md`

v0.4 关闭的是第一阶段增量 current-view 维护：

```text
S0B add/update operation
  -> S1 incremental candidate
  -> S1 delta comparison / optional LLM classification
  -> staged human review
  -> S1 current publish
  -> S2 current publish
  -> graph incremental extraction/consolidation
  -> graph current publish
  -> dependency/invalidation
  -> incremental visual review refresh
```

核心工具：

- `tools/maintenance/s0b_incremental_batch.py`
- `tools/maintenance/s1_incremental_build_runner.py`
- `tools/maintenance/s1_delta_comparison.py`
- `tools/maintenance/s1_delta_llm_classifier.py`
- `tools/maintenance/incremental_review_queue_builder.py`
- `tools/maintenance/incremental_review_webui.py`
- `tools/maintenance/incremental_review_decision_applier.py`
- `tools/maintenance/s1_current_view_publisher.py`
- `tools/maintenance/s2_current_view_publisher.py`
- `tools/maintenance/graph_incremental_extraction_runner.py`
- `tools/maintenance/graph_incremental_consolidation_runner.py`
- `tools/maintenance/graph_candidate_latest_view_runner.py`
- `tools/maintenance/graph_current_view_publisher.py`
- `tools/maintenance/graph_visual_review_incremental_refresher.py`
- `tools/maintenance/v04_full_review_workflow_package_runner.py`

人工审核已经分为三类：

```text
s1_review
s2_review
graph_review
```

审核动作简化为人能理解的一组：

```text
approve_recommended
approve_with_edit
needs_more_evidence
reject
defer
```

系统内部再把这些决策转换成 apply plan、current view overlay、refresh scope 和 invalidation。

手工 full-review workflow 验证记录：

```text
workspace: users/_v04_full_review_workflow_manual
acceptance_status: pass
review_decision_count: 3
apply_plan_count: 3
s1_current_active: 2
s2_current_active: 1
graph_changed_units: 6
visual_review_slices: 6
durable_memory_written: false
graph_truth_written: false
```

不代表：

- 物理 hard delete 已实现；
- 所有 index 自动重建；
- 实体自动合并已完成；
- 普通用户无需理解系统即可使用；
- durable memory 写入已经开放。

## 工具链盘点

### 构建 / 路由 / 索引

主要入口：

- `tools/prebuild_routing.py`
- `tools/step1/step1_build_runner.py`
- `tools/step1/step1_index_runner.py`
- `tools/step2/step2_build_runner.py`
- `tools/step2/step2_user_model_index_runner.py`

能力：

- S1/S2 pre-build route；
- proposal/provider lane；
- S1 evidence_plus_memory index；
- S2 user model embedding index；
- provider 默认真实 lane，mock 只用于测试/对照。

### 图构建 / 图查询

主要入口：

- `tools/graph/graph_construction_packet_builder.py`
- `tools/graph/graph_package_router.py`
- `tools/graph/graph_relation_candidate_extractor.py`
- `tools/graph/graph_candidate_consolidator.py`
- `tools/graph/networkx_graph_utility_runner.py`
- `tools/graph/graph_query_retriever.py`
- `tools/graph/graph_visual_review_bundle_builder.py`

能力：

- graph construction packet；
- graph-specific package routing；
- schema-guided provider extraction；
- candidate consolidation；
- NetworkX algorithm report；
- relation-neighborhood / evidence-path retrieval；
- graph-aware query package；
- visual review bundle。

### 增量维护

主要入口：

- `tools/maintenance/v04_full_review_workflow_package_runner.py`
- `tools/maintenance/v04_incremental_workflow_smoke_runner.py`
- `tools/maintenance/incremental_review_webui.py`
- `tools/maintenance/incremental_review_decision_applier.py`
- `tools/maintenance/graph_current_view_publisher.py`

能力：

- operation log；
- incremental impact resolution；
- latest/current view；
- staged human review；
- review session submit；
- S1/S2/graph current publish；
- dependency map；
- query invalidation；
- affected graph visual review refresh。

## 验证证据

已经形成的主要验收工作区：

```text
users/_acceptance_s0b_s1_independent_pepys_20260523_001842
users/_acceptance_s1_to_s2_pepys_20260522_235415
users/_acceptance_s1s2_default_provider_20260523_01
users/_v03_validation_litbank_en_20260525
users/_v04_full_review_workflow_manual
```

已记录的关键验证：

- v0.2：S0B -> S1 -> S1 Index，S1 -> S2 -> S2 Index 分段 blind acceptance。
- v0.21：真实 provider 小批 S1/S2 build/index，默认 provider lane 不再是 mock。
- v0.3：候选图 final quality gate `pass_with_known_warnings`。
- v0.31：HTML + GraphML/JSON visual review bundle，外部浏览器工具 smoke。
- v0.4：manual full-review package `acceptance_status=pass`。

最近版本管理状态：

```text
main repo: f8cb56b Document and test v0.4 incremental closure workflow
tools repo: b43d47f Add v0.4 incremental workflow closure tools
public mirror: 04b973c Add v0.4 incremental maintenance closure workflow
public GitHub: https://github.com/scy114/personal-semantic-memory-lab
```

测试记录：

- 私有主仓此前记录：全量测试 317 tests OK。
- public mirror 此前记录：全量测试 277 tests OK。

本报告生成时没有重新跑全量测试；这里引用的是现有闭环和提交记录。

## 开源成果

已经建立 public mirror：

```text
E:/code/codex/personal-semantic-memory-lab-public
```

GitHub remote：

```text
https://github.com/scy114/personal-semantic-memory-lab
```

public mirror 策略：

- 不带私有历史；
- 不公开 `users/`、`data/`、`external_references/`、真实 provider 输出、真实 `.env`；
- `tools/` 作为普通源码目录发布；
- 包含 README、LICENSE、NOTICE、SECURITY、CONTRIBUTING、公开 fixtures、精选 docs 和 CI；
- 外部资源不 vendoring，只保留资源说明。

这说明项目已经具备公开审计的最小骨架，但 public mirror 还不是“开箱即用产品”。

## 当前能力边界

可以比较有信心地说已经做到：

- 分层资产构建；
- S1/S2 路由和真实 provider assist；
- S1/S2 索引；
- 候选图构建；
- 图算法 utility proof；
- 三路 query：字面、语义、图；
- 轻量 answer context；
- 图可视化审计；
- 人工审核 WebUI；
- 增量 current view 发布；
- changed graph units 的依赖和可视化刷新；
- public mirror。

不能说已经做到：

- 可靠长期个人记忆生产系统；
- durable memory 自动写入；
- 最终画像真值；
- 最终图真值；
- S3 support checker；
- 完整实体合并自动化；
- 普通用户级产品体验；
- 大规模、多数据集、长期稳定性验证。

## 主要工程原则

这些原则已经反复写入文档和工具边界，后续不能松：

- `graph_is_not_proof=true`
- `support_status=not_checked`
- `write_permission=false`
- `visualization_is_audit_support_only=true`
- LLM output 是 candidate，不是 truth。
- Graph metric 是 ranking/retrieval signal，不是 support proof。
- Summary / description 是 helper field，不替代 evidence refs、quotes、raw backpointers。
- 非主线能力不要重复造轮子；优先使用成熟外部资源、数据集、词表、算法和工具。
- full build 和 incremental build 要保持架构分离。
- current view 是派生视图，不是 canonical history rewrite。

## 当前成熟度判断

如果按工程阶段划分，当前更像：

```text
research prototype -> engineering alpha
```

理由：

- 有真实 runner、测试、工作流、人工审核和版本管理；
- 有多个闭环验收包；
- 有 public mirror；
- 但入口仍分散，命令复杂，样例工作区和 quickstart 还不够普通用户友好；
- 真实数据上的质量还需要更多持续评估。

## v0.41 usable alpha

v0.41 的当前方向已经明确为：

```text
v0.41 usable alpha
```

目标不是继续扩理论，而是让系统更可用。第一步是把已有 tools 和 workflow 收束到统一薄入口：

```text
python -m tools.workflow_runner ...
```

当前 v0.41 已经完成 mock/synthetic acceptance path 验收：

- `status`：检查 workspace 已有哪些 S0B/S1/S2/graph/index/current-view 资产；
- `build-full`：编排 S1 Build -> S1 Index -> S2 Build -> S2 Index；
- `build-graph`：编排 graph construction、routing、extraction、consolidation、NetworkX、profile/community、quality gate 和可选 visual；
- `query`：封装 S2 query runner，并优先发现 `graph_current/`；
- `incremental`：封装 v0.4 prepare/finalize/smoke，保留 human review gate。

本轮验收记录：

```text
docs/v0.41-release-acceptance-todo.md
docs/v0.41-release-acceptance-report.md
users/_v041_release_acceptance_mock_20260530/workflow_runs/
```

下一步仍然要补的是：

1. bounded live provider smoke

   单独跑小批真实 provider 验证，不在 CI 默认运行，不把 provider 输出放入 public mirror。

2. public mirror 同步

   把 v0.41 workflow runner、测试、精选文档同步到 public mirror，但继续排除：

   ```text
   users/
   .env
   provider outputs
   external downloads
   ```

3. public-safe sample workspace

   构建一个可以公开的小样例，覆盖 full build、query、graph visual review、incremental human review、current publish。

4. README quickstart

   在 README / docs 中给出一条能跑通的最小命令链，不依赖真实 provider。

5. 可读报告入口

   把主要报告统一索引，避免每次都在几十个文档里找当前状态。

6. 质量样例集

   继续使用小而难的中文/英文样例，检查 query、图关系、实体合并、增量审核是否真的有用。

一句话：v0.41 的私有主仓入口已经验收，下一步是 live smoke 与 public mirror 可用性收尾。

## 参考文档索引

核心闭环：

- `docs/v0.2-quality-upgrade-closure-report.md`
- `docs/v0.21-build-index-closure-report.md`
- `docs/v0.3-closure-report.md`
- `docs/v0.31-graph-visualization-closure-report.md`
- `docs/v0.4-incremental-maintenance-workflow.md`
- `docs/v0.4-incremental-maintenance-todo.md`
- `docs/v0.41-workflow-toolchain-integration.md`

图系统：

- `docs/v0.3-graph-tooling-and-algorithm-research.md`
- `docs/v0.3-graphrag-core-code-learning-notes.md`
- `docs/v0.3-graph-construction-input-output-contract.md`
- `docs/v0.3-graph-toolchain-usage.md`
- `docs/references/graph/v0.3-graph-toolchain-usage.md`

增量维护：

- `docs/v0.4-incremental-maintenance-external-patterns.md`
- `tools/maintenance/v04_full_review_workflow_package_runner.py`
- `tools/maintenance/v04_incremental_workflow_smoke_runner.py`
- `tools/maintenance/incremental_review_webui.py`

公开发布：

- `E:/code/codex/personal-semantic-memory-lab-public/README.md`
- `E:/code/codex/personal-semantic-memory-lab-public/OPEN_SOURCE_AUDIT.md`
- `E:/code/codex/personal-semantic-memory-lab-public/PUBLIC_RELEASE_TODO.md`
