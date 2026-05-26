# User-rooted Graph Summary - Jon

生成日期：2026-05-16

范围：公开数据集 LoCoMO `conv-30`，目标参与者 `Jon`，当前只覆盖 sessions 1-5。

状态：Step 2 pilot 的第一版 user-rooted graph。图谱只表达结构关系，不替代 evidence，不替代 retrieval ranking，也不代表真实用户确认。

## 1. 图谱规模

- 节点数：24
- 边数：22
- 边状态：
  - `active`: 16
  - `candidate`: 6
- 存储方式：JSONL / Markdown，不使用图数据库。

## 2. 核心结构

这张图以 `public_locomo_conv_30_jon` 作为根节点，围绕 Jon 的舞蹈事业和个人价值组织关系。

主要子结构：

- `dance_studio_project`: Jon 当前最核心的 project，连接 studio location、site requirements、inclusive expression goal、business/dance balancing 等节点。
- `dance_interest`: Jon 的长期兴趣和 passion，支持画像层对“舞蹈不是短期兴趣”的判断。
- `contemporary_dance_preference`: Jon 明确表达的舞蹈偏好。
- `gina_support_relationship`: Gina 作为支持关系对象存在；她不是 Jon 的画像主体。
- `risk_taking_value` 与 `freedom_self_expression_value`: Jon 的价值倾向，分别对应冒险成长、自由表达。

## 3. 关系类型分布

本轮生成的边主要包括：

- `works_on`: Jon 与 dance studio / festival / competition 等项目或事件的关系。
- `has_goal`: dance studio project 与理想地点、inspiring space、right location、inclusive expression 等目标的关系。
- `has_constraint`: dance studio project 与 site requirements、balancing dance and business 等约束的关系。
- `has_routine`: Jon 与 rehearsal / business work routine 的关系。
- `has_procedure`: Jon 面对鼓励和障碍时的行动方式。
- `has_value`: Jon 的风险、自由、自我表达等价值倾向。
- `has_relationship`: Jon 与 Gina 支持关系的结构化表达。
- `indicates`: 从 setback difficulty 指向 need encouragement，保留为推断候选。
- `produced_lesson`: 从 competition win 指向 performance confidence，保留为候选。

其中 `has_state` 已规范化为文档中已有的 `updates`，用于表达 `dance_studio_project -> studio_location_search` 的状态更新。

## 4. Active 与 Candidate 的含义

本轮规则：

- 高置信、显式证据、非推断关系：设为 `active`。
- 中置信、强推断、`indicates`、`produced_lesson`：设为 `candidate`。

这意味着图谱不是把所有关系都当成事实，而是保留了一层不确定性。

## 5. 需要后续人工关注的点

- `dance_studio_project` 的时间线仍然不完全稳定：有些证据像是在寻找地点，有些证据像是已经在运行 studio。
- festival、upcoming show、nearby competition 可能是不同事件，也可能有重叠，暂不合并。
- `need_encouragement` 是从 setback 语境中推断出的候选节点，不能当作稳定事实。
- `performance_confidence` 也是候选 lesson，后续需要更多证据或任务表现验证。

## 6. 下一步用途

下一步可以基于这张图生成 `base_assistance_packet.json`：

- 从 portrait 读取 Jon 的稳定画像。
- 从 graph 扩展相关 project、goal、constraint、routine、relationship。
- 从 evidence refs 回溯事实来源。
- 给后续 task-centered evaluation 提供“画像 + 图 + 证据”的输入。
