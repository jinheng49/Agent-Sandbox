# Project Darwin

Project Darwin 是一个多智能体生存沙盒，用来观察资源压力、通信成本、协作行为和代际学习接口在统一离散回合环境中的演化趋势。

当前仓库已经完成以下基础能力：

- 一个可运行的 10x10 回合制世界。
- 多个带不同性格标签的 Agent。
- 资源采集、移动、消息计费、死亡规则。
- 统一事件流与逐回合快照。
- Replay 文件落盘与 Dashboard 历史帧回看。
- 启发式本能层与基础 gold 协作机制。
- 本地 lineage 记忆层、规则反思提取与向量检索。
- Qdrant 向量化存储接入、run-group 隔离与结构化记忆检索。
- LLM 决策层、第三方 API 适配、结构化动作输出与启发式兜底回退。

当前项目重点不是“已经拥有完整 LLM Agent”，而是已经搭好了一个可复现、可回放、可扩展的实验底座，后续可以继续往欺骗机制、信任状态、LLM 决策层和自动化代际实验推进。

## 1. 当前项目目标

Project Darwin 当前关注的问题是：

- Agent 在资源稀缺和通信收费条件下会如何行动。
- 不同性格标签是否会诱发不同的资源竞争与协作模式。
- Gold 这样的高价值资源是否会触发协作、独占或后续的误导行为。
- 如何通过统一事件流、快照和 replay 为后续记忆与演化实验提供数据基础。

## 2. 当前世界规则

### 2.1 地图与资源

- 世界默认是 10x10 网格。
- 地图上有两种资源：
  - `food`：普通资源。
  - `gold`：高价值资源。
- 初始资源分布由确定性规则生成，而不是完全随机。这保证了实验可复现。

相关实现：

- [project_darwin/environment/resource_rules.py](project_darwin/environment/resource_rules.py)
- [project_darwin/simulation/run_context.py](project_darwin/simulation/run_context.py)

### 2.2 Agent 可执行动作

当前 Agent 只能通过结构化动作影响世界，不允许直接自由文本控制环境。

动作类型包括：

- `rest`
- `move`
- `message`
- `forage`

相关实现：

- [project_darwin/agents/action_space.py](project_darwin/agents/action_space.py)

### 2.3 能量与生存规则

- 移动会扣除固定能量。
- 发消息会按字符长度扣除能量。
- 采集 `food` 或 `gold` 会增加能量。
- 如果 Agent 能量小于等于 0，则死亡。

- `sender_id`
- `sender_family_id`
- `content`
目前已支持的消息意图包括：

- `contact`
- `share_food`
- `share_gold`

这为后续做协作、误导、欺骗检测提供了基础。

相关实现：

- [project_darwin/agents/action_space.py](project_darwin/agents/action_space.py)
- [project_darwin/environment/env_engine.py](project_darwin/environment/env_engine.py)
- 合作分账会产生独立的 `cooperation` 事件。

相关实现：
- [project_darwin/environment/env_engine.py](project_darwin/environment/env_engine.py)

## 3. 当前 Agent 类型

### 3.1 ScriptedSurvivor
- 看到附近 Agent 时，非 silent 会发极简 token。
- 其他情况按固定方向循环移动。


- [project_darwin/agents/base_agent.py](project_darwin/agents/base_agent.py)

### 3.2 RandomSurvivor

这是受约束随机 Agent，主要用于验证逻辑覆盖：


相关实现：

这是当前阶段 1 的核心 Agent，也是“启发式本能层”的实现。

其决策链是：

- 读取 trait config。

相关实现：

- [project_darwin/agents/heuristic_agent.py](project_darwin/agents/heuristic_agent.py)
- [project_darwin/agents/policy.py](project_darwin/agents/policy.py)
- [project_darwin/agents/traits.py](project_darwin/agents/traits.py)

## 4. 启发式本能层说明

阶段 1 的目标是：在不调用大模型的情况下，先跑出稳定、可解释的性格差异。

### 4.1 Trait 层

当前 trait 体系由两部分组成：

- `TraitProfile`：身份标签。
- `TraitConfig`：行为偏置配置。

每种 trait 当前都配置了：

- 基础动作偏置。
- 低能量偏置。
- 看到 food 的偏置。
- 看到 gold 的偏置。
- 看到其他 Agent 的偏置。
- 资源优先级。
- 探索方向循环。
- 默认消息 token。

目前三种 trait 为：

- `GREEDY`
- `COOPERATIVE`
- `SILENT`

### 4.2 Policy 层

当前 policy 层主要做三件事：

- 把 observation 压成一组可解释特征。
- 根据 TraitConfig 计算动作分数。
- 选择移动方向与动作采样。

核心特征目前包括：

- 是否低能量。
- 脚下是否有资源。
- 视野内 food 数量。
- 视野内 gold 数量。
- 附近 Agent 数量。

## 5. 事件系统与 Replay

### 5.1 统一事件 Schema

阶段 0 已经完成统一事件结构。

每条事件至少包含：

- `turn`
- `event_type`
- `agent_id`
- `family_id`
- `payload`

当前 event_type 规范包括：

- `turn_start`
- `turn_end`
- `move`
- `message`
- `forage`
- `forage_miss`
- `cooperation`
- `rest`
- `death`

相关实现：

- [project_darwin/simulation/event_bus.py](project_darwin/simulation/event_bus.py)

### 5.2 逐回合快照

每次运行现在都会保存 `snapshots`，每一帧都能恢复：

- 当前 turn。
- 全部 Agent 状态。
- 全部资源状态。
- 当前 event_count。

这使得 Dashboard 能够做历史帧回看，而不只是展示最终结果。

相关实现：

- [project_darwin/simulation/state.py](project_darwin/simulation/state.py)
- [project_darwin/simulation/scheduler.py](project_darwin/simulation/scheduler.py)
- [project_darwin/analytics/replay_store.py](project_darwin/analytics/replay_store.py)

### 5.3 Replay 元数据

每个 replay 文件目前包含：

- `metadata`
- `world`
- `events`
- `snapshots`
- `metrics`
- `communication`

`metadata` 当前至少包括：

- `run_id`
- `generation`
- `lineage_id`
- `mode`
- `seed`
- `family_ids`

## 6. 当前分析能力

## 6. 记忆层与分析能力

### 6.1 Lineage Memory

阶段 2 已完成一个本地可运行的记忆层，目标不是直接接入 LLM 总结，而是先把可验证的数据闭环跑通。

当前流程是：

- 运行结束后，`reflection_engine.py` 会从死亡事件及最近事件窗口中提取规则化反思。
- 反思被写入本地嵌入式 Qdrant，并按 `experiment_id + run_group` 隔离存放在 `artifacts/qdrant/`。
- 后续回合中，`scheduler.py` 会在 agent 决策前根据当前 observation 检索同 family、同 lineage 的结构化记忆结果，再把 lesson 文本映射给 agent。
- `HeuristicSurvivor` 会把检索结果作为 memory context，对明显危险的高成本行为做轻量降权。

当前记忆条目包含：

 - `experiment_id`
 - `run_group`
- `family_id`
- `lineage_id`
- `generation`
 - `trait`
 - `death_reason`
 - `memory_type`
- `source_run_id`
- `source_agent_id`
- `death_turn`
- `situation`
- `lesson`
- `tags`

当前检索结果至少包含：

- memory text
- lesson
- score
- generation
- metadata

相关实现：

- [project_darwin/memory/lineage_store.py](project_darwin/memory/lineage_store.py)
- [project_darwin/memory/reflection_engine.py](project_darwin/memory/reflection_engine.py)
- [project_darwin/memory/retrieval_engine.py](project_darwin/memory/retrieval_engine.py)
- [project_darwin/simulation/scheduler.py](project_darwin/simulation/scheduler.py)
- [project_darwin/agents/heuristic_agent.py](project_darwin/agents/heuristic_agent.py)

### 6.2 Metrics

当前统计项包括：

- 当前 turn
- 存活 Agent 数
- 总消息数
- 总消息成本
- 采集次数
- 协作事件次数
- `false_gold` 消息数
- trust 更新次数
- 剩余资源数

相关实现：

- [project_darwin/analytics/metrics_engine.py](project_darwin/analytics/metrics_engine.py)

### 6.3 Communication Analysis

当前通信分析包括：

- 词表大小
- 平均消息长度
- 信息熵
- `share_gold` 信号数
- `false_gold` 信号数

相关实现：

- [project_darwin/analytics/communication_analysis.py](project_darwin/analytics/communication_analysis.py)

## 7. Dashboard 当前能力

当前 Dashboard 基于 Streamlit，已经可以：

- 直接触发模拟运行。
- 选择运行模式：`scripted`、`random`、`heuristic`。
- 加载已有 replay。
- 使用 Replay Frame 滑块查看历史帧。
- 查看地图、Agent 状态、Agent 日志。
- 查看 Global Event Stream 与当前回合事件。
- 查看 Timeline Mode。
- 查看 Replay Summary。

相关实现：

- [project_darwin/dashboard/app.py](project_darwin/dashboard/app.py)

## 8. 项目结构说明

```text
project_darwin/
├── agents/
│   ├── action_space.py
│   ├── base_agent.py
│   ├── cognition_graph.py
│   ├── heuristic_agent.py
│   ├── llm_adapter.py
│   ├── policy.py
│   ├── random_agent.py
│   └── traits.py
├── analytics/
│   ├── communication_analysis.py
│   ├── metrics_engine.py
│   └── replay_store.py
├── dashboard/
│   ├── app.py
│   ├── data_reader.py
│   └── terminal_view.py
├── environment/
│   ├── env_engine.py
│   ├── observation_builder.py
│   └── resource_rules.py
├── experiments/
│   ├── configs/default.py
│   ├── run_manager.py
│   └── terminal_run.py
├── memory/
│   ├── lineage_store.py
│   ├── reflection_engine.py
│   └── retrieval_engine.py
└── simulation/
    ├── event_bus.py
    ├── run_context.py
    ├── scheduler.py
    └── state.py
```

### 8.1 agents

- `action_space.py`：动作和消息结构。
- `base_agent.py`：BaseAgent 与 ScriptedSurvivor。
- `heuristic_agent.py`：阶段 1 的启发式 Agent。
- `random_agent.py`：随机 Agent。
- `traits.py`：trait 标签与配置库。
- `policy.py`：启发式策略层。
- `llm_adapter.py`：大模型接口占位层。
- `cognition_graph.py`：LangGraph 认知图占位层。

### 8.2 analytics

- `metrics_engine.py`：全局指标统计。
- `communication_analysis.py`：消息分析。
- `replay_store.py`：replay 落盘。

### 8.3 dashboard

- `app.py`：网页控制台与回放页面。
- `data_reader.py`：读取 replay 文件。
- `terminal_view.py`：终端渲染器。

### 8.4 environment

- `env_engine.py`：执行动作并修改世界。
- `observation_builder.py`：构建 Agent 观测。
- `resource_rules.py`：初始资源布局。

### 8.5 experiments

- `configs/default.py`：默认实验参数。
- `run_manager.py`：组装 world 与 agents。
- `terminal_run.py`：终端模式入口。

### 8.6 memory

- `lineage_store.py`：本地嵌入式 Qdrant lineage 记忆仓库。
- `reflection_engine.py`：基于死亡与最近事件窗口的规则反思提取。
- `retrieval_engine.py`：基于 observation 的同族记忆检索接口。

### 8.7 simulation

- `event_bus.py`：统一事件系统。
- `run_context.py`：SimulationConfig。
- `scheduler.py`：核心调度器。
- `state.py`：世界状态与序列化。

## 9. 一次运行的调用链

点击 Dashboard 中的 `Run Live Simulation` 后，主调用链是：

1. [project_darwin/dashboard/app.py](project_darwin/dashboard/app.py) `main()`
2. [project_darwin/dashboard/app.py](project_darwin/dashboard/app.py) `_run_live_simulation(mode)`
3. [project_darwin/experiments/configs/default.py](project_darwin/experiments/configs/default.py) `default_config()`
4. [project_darwin/experiments/run_manager.py](project_darwin/experiments/run_manager.py) `build_*_simulation(...)`
5. [project_darwin/simulation/scheduler.py](project_darwin/simulation/scheduler.py) `Scheduler.run(...)`
7. [project_darwin/environment/observation_builder.py](project_darwin/environment/observation_builder.py) `build_observation(...)`
8. [project_darwin/memory/retrieval_engine.py](project_darwin/memory/retrieval_engine.py) 检索当前 observation 对应的 lineage memories
9. Agent `set_memory_context(...)` 与 `choose_action(...)`
10. [project_darwin/environment/env_engine.py](project_darwin/environment/env_engine.py) `step(...)`
11. [project_darwin/simulation/event_bus.py](project_darwin/simulation/event_bus.py) 记录事件
12. [project_darwin/analytics/metrics_engine.py](project_darwin/analytics/metrics_engine.py) 和 [project_darwin/analytics/communication_analysis.py](project_darwin/analytics/communication_analysis.py)
13. [project_darwin/memory/reflection_engine.py](project_darwin/memory/reflection_engine.py) 提取死亡反思
14. [project_darwin/memory/lineage_store.py](project_darwin/memory/lineage_store.py) 写入本地记忆库
15. [project_darwin/analytics/replay_store.py](project_darwin/analytics/replay_store.py) `write_run(...)`
16. [project_darwin/dashboard/data_reader.py](project_darwin/dashboard/data_reader.py) `load_replay(...)`

## 10. 当前项目还没完成的部分

以下能力目前仍处于占位或半成品状态：

- LLM API 调用尚未接入。
- LangGraph 认知图尚未接入真实流程。
- 反思仍是规则抽取，不是 LLM 生成总结。
- 记忆检索当前使用本地哈希嵌入，不是语义质量更高的模型 embedding。
- 欺骗与误导检测还未实现。
- 信任关系、背叛、家族策略传承还未实现。
- gold 协作规则目前只是第一版，不含谈判、争夺和信誉成本。

## 11. 当前开发阶段状态

### 已完成

- 阶段 0：事件 schema、回合快照、run metadata、结构化消息底座。
- 阶段 1：启发式本能层。
- 阶段 1.5：gold 协作规则、语义化消息意图、基础博弈事件统计。
- 阶段 2：本地 lineage 记忆库、规则反思提取、同族检索与记忆注入。
- 阶段 3：欺骗消息、短期 trust tracker、基于 social hint 的移动偏置。
- 阶段 4：Qdrant 向量化接入、run-group 隔离、结构化检索返回与阈值参数化。
- 阶段 5：第三方 LLM API 接入、专属 LLM Agent、结构化动作输出与防御性回退。

### 下一步建议

更自然的下一阶段是：

1. 把 trust 状态接进 Dashboard 和 replay 详情页。
2. 扩展更多欺骗策略与信誉成本。
3. 用真实 embedding 与 LLM reflection 替换当前规则版 memory pipeline。

### 12.5 配置第三方 LLM API

当前 LLM 层使用 OpenAI 兼容接口，优先读取以下环境变量：

- `DARWIN_LLM_API_KEY`
- `DARWIN_LLM_BASE_URL`
- `DARWIN_LLM_MODEL`

示例：

```bash
export DARWIN_LLM_API_KEY="your-third-party-key"
export DARWIN_LLM_BASE_URL="https://your-provider.example/v1"
export DARWIN_LLM_MODEL="your-model-name"
```

仓库根目录也提供了一个可直接修改的配置文件：[.env.llm](/workspaces/jinheng/project-Darwin/.env.llm)。

加载方式：

```bash
set -a
source .env.llm
set +a
```

## 12. 运行方式

### 12.1 运行测试

```bash
python -m unittest discover -s tests
```

### 12.2 运行命令行主入口

```bash
python -m project_darwin.experiments.run_manager
```

### 12.3 运行终端模式

```bash
python -m project_darwin.experiments.terminal_run
```

### 12.4 启动 Streamlit Dashboard

```bash
streamlit run project_darwin/dashboard/app.py --server.address 0.0.0.0 --server.port 8502
```

## 13. 测试覆盖范围

当前测试已覆盖：

- replay 与 snapshot 是否正确落盘。
- event schema 是否完整。
- message 是否为结构化消息体。
- random simulation 是否能产生消息。
- heuristic simulation 是否能运行。
- heuristic trait 配置是否完整。
- cooperative gold 消息是否结构化。
- gold 是否能与附近 Agent 协作分账。
- 阶段 2 反思是否会落入 lineage store。
- 同 family 的记忆是否能被检索，异族记忆是否被过滤。
- 不同 run_group 的 Qdrant 记忆是否物理隔离。
- retrieval engine 是否返回结构化 memory 结果与必要 metadata。
- heuristic agent 是否接收 memory context 并调整 message 倾向。
- 非法或残缺的模型输出是否会触发修复重试与 heuristic fallback。
- llm agent 是否能与 scripted agent 同场运行。

相关文件：

- [tests/test_simulation.py](tests/test_simulation.py)

## 14. 项目定位总结

当前的 Project Darwin 是：

- 一个已经能稳定推进世界状态的回合制沙盒。
- 一个已经具备统一事件流和 replay 回看的实验平台。
- 一个已经开始出现启发式性格差异与基础协作行为的 Agent 系统。

它还不是完整的“多代演化 LLM 社会”，但已经具备继续向那个方向推进所需的工程底座。