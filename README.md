# Sandbox

Sandbox 是一个多智能体生存沙盒，用于做可复现的 agent 行为实验。当前仓库已经不只是基础移动和采集原型，而是一个包含离散回合制环境、回放系统、结构化记忆检索、短期规划、社会推理、LLM 决策、基准实验和可视化工作台的实验底座。

当前实现更适合回答这类问题：

- 资源压力、通信成本和地图尺度会怎样改变 agent 的策略。
- 带记忆、规划、社会推理的 LLM agent 是否优于弱化版本或启发式基线。
- 协作、欺骗、信任更新和资源竞争会在什么条件下更频繁出现。
- 不同实验配置下，结果能否被稳定回放、比较、聚合和归档。

> 兼容说明：项目展示名已更新为 Sandbox，但当前 Python 包名仍然是 `project_darwin`，LLM 环境变量前缀仍然是 `DARWIN_LLM_*`。README 下面所有命令均以现有代码接口为准。

## 当前能力

- 确定性的离散回合制网格世界。
- 默认 6x6 方形地图，Dashboard 中可调 5 到 12 的边长。
- 结构化动作空间：移动、采集、发送消息、休息。
- 统一事件流、逐回合快照、Replay 持久化与复盘。
- Scripted、Random、Heuristic、LLM 四类 agent。
- 基于 family 和 lineage 的本地 Qdrant 记忆存储与检索。
- 结构化记忆注入：hard constraints、soft hints、examples、typed lessons。
- 短期规划：goal 和 planned target 会进入决策与回放。
- 显式社会推理：reputation、utility、alliance likelihood、threat level。
- 基准实验与消融实验：memory、planning、social reasoning 可开关。
- Streamlit Dashboard，支持模拟模式与复盘模式。

## 核心机制

### 世界与资源

- 世界是方形网格地图，默认配置为 6x6。
- 资源包括 food 和 gold。
- 初始资源分布由确定性规则生成，保证实验可复现。
- 当前默认配置没有启用集中式资源布局，竞争主要通过小地图和资源压力触发。

相关实现：

- [project_darwin/simulation/run_context.py](project_darwin/simulation/run_context.py)
- [project_darwin/environment/resource_rules.py](project_darwin/environment/resource_rules.py)
- [project_darwin/experiments/configs/default.py](project_darwin/experiments/configs/default.py)

### 动作空间

Agent 只能通过结构化动作影响环境，当前动作包括：

- rest
- move
- forage
- message

动作对象还会记录：

- decision_source
- decision_note
- current_goal
- planned_target_position

相关实现：

- [project_darwin/agents/action_space.py](project_darwin/agents/action_space.py)

### 能量、采集与死亡

- 移动会扣除固定能量。
- 发送消息会按字符长度扣除能量。
- 采集 food 或 gold 会增加能量。
- 能量小于等于 0 时 agent 死亡。
- 采集时可能触发 cooperation 事件。

相关实现：

- [project_darwin/environment/env_engine.py](project_darwin/environment/env_engine.py)

### 终止条件

- 默认情况下，只剩 1 个 agent 存活时会提前结束。
- 同时始终保留最大轮次作为上限。
- Dashboard 可显式关闭提前结束逻辑。

相关实现：

- [project_darwin/simulation/scheduler.py](project_darwin/simulation/scheduler.py)
- [project_darwin/simulation/run_context.py](project_darwin/simulation/run_context.py)

## Agent 类型

### ScriptedSurvivor

规则最简单的 agent，用于生成稳定、可解释的基线行为：

- 有资源就采集。
- 在某些条件下发送极简消息。
- 其他情况按固定模式移动。

相关实现：

- [project_darwin/agents/base_agent.py](project_darwin/agents/base_agent.py)

### RandomSurvivor

受约束随机 agent，主要用于压测环境和事件链路。

相关实现：

- [project_darwin/agents/random_agent.py](project_darwin/agents/random_agent.py)

### HeuristicSurvivor

当前最强的非 LLM agent，也是 LLM 失败时的 fallback。它会联合使用：

- trait 偏置
- 扩展 observation
- 结构化 memory package
- 当前短期计划
- 资源热点和未探索位置
- 社会推理结果和可疑信号过滤

相关实现：

- [project_darwin/agents/heuristic_agent.py](project_darwin/agents/heuristic_agent.py)
- [project_darwin/agents/policy.py](project_darwin/agents/policy.py)
- [project_darwin/agents/traits.py](project_darwin/agents/traits.py)

### LLMSurvivor

LLM agent 已接入 OpenAI 兼容接口，当前能力包括：

- 使用结构化 prompt 做动作决策。
- 接收结构化记忆包而不是平面字符串。
- 接收当前 short-term plan。
- 接收显式社会推理上下文。
- 对输出进行结构化解析和防御性校验。
- 在模型不可用或返回异常时回退到 HeuristicSurvivor。

相关实现：

- [project_darwin/agents/llm_agent.py](project_darwin/agents/llm_agent.py)
- [project_darwin/agents/llm_adapter.py](project_darwin/agents/llm_adapter.py)
- [project_darwin/agents/cognition_graph.py](project_darwin/agents/cognition_graph.py)
- [project_darwin/agents/prompt_builder.py](project_darwin/agents/prompt_builder.py)

## Observation、Memory、Planning、Social Reasoning

### 扩展 Observation

当前 observation 除了可见资源和附近 agent，还包括：

- recent self events
- recent received messages
- recent positions
- explored positions
- nearby unexplored positions
- resource hotspots
- agent social profiles
- exploration ratio

相关实现：

- [project_darwin/environment/observation_builder.py](project_darwin/environment/observation_builder.py)
- [project_darwin/simulation/scheduler.py](project_darwin/simulation/scheduler.py)

### Case-Based Memory

当前记忆包含四类 case：

- death_reflection
- success_reflection
- cooperation_reflection
- deception_reflection

反思会在运行结束后生成并写入本地 store，之后按 family 和 lineage 检索。

相关实现：

- [project_darwin/memory/reflection_engine.py](project_darwin/memory/reflection_engine.py)
- [project_darwin/memory/lineage_store.py](project_darwin/memory/lineage_store.py)

### Structured Retrieval

检索结果会整理成 MemoryContextPackage，当前包结构包括：

- hard_constraints
- soft_hints
- examples
- typed_lessons
- directives，例如 action_bias、target_preference、caution_against

相关实现：

- [project_darwin/memory/retrieval_engine.py](project_darwin/memory/retrieval_engine.py)

### Short-Term Planning

当前系统支持短期计划，计划会影响 heuristic 和 LLM 的决策，并写入 replay：

- current_goal
- planned_target_position
- created_turn

相关实现：

- [project_darwin/agents/action_space.py](project_darwin/agents/action_space.py)
- [project_darwin/agents/llm_agent.py](project_darwin/agents/llm_agent.py)
- [project_darwin/agents/prompt_builder.py](project_darwin/agents/prompt_builder.py)
- [project_darwin/environment/env_engine.py](project_darwin/environment/env_engine.py)

### Social Reasoning

当前社会推理显式维护并暴露：

- sender_reputation
- message_utility
- alliance_likelihood
- threat_level

这些值会进入 observation、heuristic 决策和 LLM prompt。

相关实现：

- [project_darwin/simulation/trust_tracker.py](project_darwin/simulation/trust_tracker.py)
- [project_darwin/environment/observation_builder.py](project_darwin/environment/observation_builder.py)
- [project_darwin/agents/heuristic_agent.py](project_darwin/agents/heuristic_agent.py)
- [project_darwin/agents/prompt_builder.py](project_darwin/agents/prompt_builder.py)

## Replay、指标与实验输出

### 统一事件流

系统会记录结构化事件，常见事件包括：

- turn_start
- turn_end
- move
- message
- forage
- forage_miss
- cooperation
- rest
- death
- trust_update

相关实现：

- [project_darwin/simulation/event_bus.py](project_darwin/simulation/event_bus.py)

### Replay 与快照

每次运行都会写出：

- metadata
- world
- events
- snapshots
- metrics
- communication
- run_summary

单次运行默认输出到 `artifacts/latest_run.json`。归档运行会输出到 `artifacts/<experiment>/<run_group>/.../replay.json`。Dashboard 会自动扫描 `artifacts` 目录下的 replay 和 manifest。

相关实现：

- [project_darwin/analytics/replay_store.py](project_darwin/analytics/replay_store.py)
- [project_darwin/dashboard/data_reader.py](project_darwin/dashboard/data_reader.py)
- [project_darwin/simulation/state.py](project_darwin/simulation/state.py)
- [project_darwin/simulation/scheduler.py](project_darwin/simulation/scheduler.py)

### Metrics 与通信分析

当前指标包括：

- alive_agents
- average_survival_turn
- total_messages
- total_message_cost
- total_forage_events
- total_cooperation_events
- total_false_gold_messages
- total_trust_updates
- resource_acquisition_rate
- message_cost_per_turn
- cooperation_rate
- deception_frequency

通信分析包括：

- vocabulary_size
- mean_message_length
- entropy

相关实现：

- [project_darwin/analytics/metrics_engine.py](project_darwin/analytics/metrics_engine.py)
- [project_darwin/analytics/communication_analysis.py](project_darwin/analytics/communication_analysis.py)

## Benchmark 与消融实验

当前实验入口支持：

- 单次运行。
- 多 generation、多 run 的归档实验。
- baseline benchmark。
- extended benchmark。
- 关闭 planning 或 social reasoning 的能力开关。

当前 benchmark group 包括：

- scripted
- heuristic
- llm
- llm_without_memory
- llm_with_memory
- llm_without_planning
- llm_without_social_reasoning

相关实现：

- [project_darwin/experiments/run_manager.py](project_darwin/experiments/run_manager.py)
- [project_darwin/analytics/metrics_engine.py](project_darwin/analytics/metrics_engine.py)

## Dashboard

Dashboard 基于 Streamlit，支持两种工作模式：

- 模拟模式
- 复盘模式

模拟模式支持：

- 选择 agent 模式：scripted、random、heuristic、llm
- 调整方形地图边长
- 调整最大轮次
- 配置是否在只剩 1 个 agent 时提前结束
- 调整步进渲染延迟
- 检查当前 LLM 是否已配置

复盘模式支持：

- 自动发现 artifacts 下的 replay
- 选择已有 replay
- 按帧前进、后退、跳到终局
- 查看地图、当前回合动作和消息
- 查看计划、决策来源、社会线索等回放信息
- 查看实验 manifest 的按代汇总

相关实现：

- [project_darwin/dashboard/app.py](project_darwin/dashboard/app.py)
- [project_darwin/dashboard/data_reader.py](project_darwin/dashboard/data_reader.py)

## 快速开始

### 1. 安装依赖

仓库当前没有单独的依赖清单文件，Dockerfile 中安装的依赖如下：

```bash
python -m pip install --upgrade pip
python -m pip install langgraph langchain-core qdrant-client streamlit pydantic rich openai
```

如果你直接使用仓库中的容器环境，默认就是 Python 3.11。

### 2. 运行测试

```bash
python -m unittest discover -s tests -q
```

### 3. 运行单次模拟

```bash
python -m project_darwin.experiments.run_manager --mode heuristic --max-turns 25
```

可选模式：`scripted`、`random`、`heuristic`、`llm`。

### 4. 运行归档实验

```bash
python -m project_darwin.experiments.run_manager \
	--mode llm \
	--experiment-id sandbox_experiment \
	--run-group local \
	--generations 2 \
	--runs-per-generation 3 \
	--archive-experiment
```

如果 `--generations` 大于 1 或 `--runs-per-generation` 大于 1，代码会自动切到归档实验路径，即使不显式传 `--archive-experiment`。

### 5. 运行 benchmark

基础 benchmark：

```bash
python -m project_darwin.experiments.run_manager --benchmark-baselines --runs-per-group 30
```

扩展 benchmark：

```bash
python -m project_darwin.experiments.run_manager --benchmark-baselines --benchmark-extended --runs-per-group 30
```

### 6. 运行终端视图

```bash
python -m project_darwin.experiments.terminal_run
```

### 7. 启动 Dashboard

```bash
streamlit run project_darwin/dashboard/app.py --server.address 0.0.0.0 --server.port 8502
```

## LLM 配置

LLM 层使用 OpenAI 兼容接口，适配器会优先读取以下环境变量：

- DARWIN_LLM_API_KEY
- DARWIN_LLM_BASE_URL
- DARWIN_LLM_MODEL

同时兼容以下 OpenAI 风格变量：

- OPENAI_API_KEY
- OPENAI_BASE_URL
- OPENAI_MODEL

仓库根目录的 [.env.llm](.env.llm) 会由适配器自动读取，因此可以直接维护这个文件。

示例：

```bash
export DARWIN_LLM_API_KEY="your-key"
export DARWIN_LLM_BASE_URL="https://your-provider.example/v1"
export DARWIN_LLM_MODEL="your-model-name"
```

如果没有配置可用 API key，LLM agent 会退回到 heuristic fallback，而不是直接让运行失败。

## 项目结构

```text
project_darwin/
├── agents/
├── analytics/
├── dashboard/
├── environment/
├── experiments/
├── memory/
└── simulation/
```

主要目录职责：

- agents：动作结构、启发式 agent、LLM agent、prompt 和认知图。
- analytics：指标、通信分析、replay 输出。
- dashboard：Streamlit 工作台与终端视图。
- environment：环境执行、资源规则、observation 构造。
- experiments：默认配置、批量实验和命令行入口。
- memory：反思生成、检索、存储。
- simulation：状态、事件总线、调度器、运行配置。

## 当前测试覆盖

当前测试已经覆盖到：

- replay 和 snapshots 是否正确落盘
- event schema 是否完整
- 并行收集动作是否生效
- 扩展 observation 是否正确注入
- case-based memory 是否生成与检索
- structured memory package 是否进入 agent 和 prompt
- 短期规划是否更新并影响 fallback
- social reasoning 是否进入 observation 和 agent profile
- benchmark group 与 planning 或 social toggles 是否正确记录
- 默认小地图配置是否生效
- 终止条件是否支持只剩 1 个存活者时提前结束

## 更自然的后续方向

1. 把 replay 页面上的运行参数和 toggle 展示得更完整。
2. 让 Dashboard 暴露更多实验参数，而不只是地图和终止条件。
3. 继续增强竞争机制，例如调整出生点或资源刷新规则。