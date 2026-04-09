import json
import tempfile
import time
import unittest
from pathlib import Path

from project_darwin.agents.action_space import ActionType, AgentAction, Direction, MessageIntent, ShortTermPlan
from project_darwin.agents.base_agent import BaseAgent, ScriptedSurvivor
from project_darwin.agents.cognition_graph import CognitionGraph
from project_darwin.agents.heuristic_agent import HeuristicSurvivor
from project_darwin.agents.llm_agent import LLMSurvivor
from project_darwin.agents.policy import get_action_scores
from project_darwin.agents.prompt_builder import build_llm_prompts
from project_darwin.agents.traits import TraitProfile, get_trait_config
from project_darwin.analytics.communication_analysis import CommunicationAnalysis
from project_darwin.environment.env_engine import EnvironmentEngine
from project_darwin.environment.resource_rules import build_initial_resources
from project_darwin.environment.observation_builder import (
    AgentSocialProfile,
    RecentEventSummary,
    ReceivedMessageSummary,
    ResourceHotspot,
    SocialHint,
    build_observation,
)
from project_darwin.experiments.run_manager import (
    build_heuristic_simulation,
    build_llm_simulation,
    build_random_simulation,
    build_scripted_simulation,
    run_baseline_benchmark,
    run_generational_experiment,
)
from project_darwin.simulation.event_bus import EventBus
from project_darwin.simulation.event_bus import EventType
from project_darwin.memory.lineage_store import MemoryRecord
from project_darwin.memory.reflection_engine import ReflectionEngine
from project_darwin.memory.retrieval_engine import MemoryContextPackage, MemoryDirective, RetrievalEngine
from project_darwin.simulation.run_context import SimulationConfig
from project_darwin.simulation.scheduler import Scheduler
from project_darwin.simulation.state import AgentState, Position, ResourceNode, ResourceType, WorldState, deserialize_world_state
from project_darwin.simulation.trust_tracker import TrustTracker
from project_darwin.experiments.configs.default import default_config
from project_darwin.dashboard.data_reader import build_experiment_catalog, build_generation_metric_rows


class FakeLLMAdapter:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            raise RuntimeError("No fake responses left")
        return self.responses.pop(0)


class SlowAgent(BaseAgent):
    def __init__(self, *args, action_log: list[tuple[str, str, float]], sleep_seconds: float = 0.12, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_log = action_log
        self.sleep_seconds = sleep_seconds

    def choose_action(self, observation):
        self.action_log.append((self.agent_id, "start", time.perf_counter()))
        time.sleep(self.sleep_seconds)
        self.action_log.append((self.agent_id, "end", time.perf_counter()))
        return AgentAction(action_type=ActionType.REST)


class SimulationTestCase(unittest.TestCase):
    def test_scheduler_produces_replay_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=6, artifact_dir=Path(temp_dir))
            world, agents = build_scripted_simulation(config)
            scheduler = Scheduler(config)

            result = scheduler.run(world, agents, replay_name="test_run.json")

            self.assertEqual(result.metrics.turn, 6)
            self.assertTrue(result.replay_path.exists())

            payload = json.loads(result.replay_path.read_text(encoding="utf-8"))
            self.assertIn("metadata", payload)
            self.assertIn("metrics", payload)
            self.assertIn("communication", payload)
            self.assertIn("snapshots", payload)
            self.assertEqual(payload["metadata"]["mode"], "scripted")
            self.assertEqual(payload["snapshots"][0]["turn"], 0)
            self.assertEqual(payload["snapshots"][-1]["turn"], 6)

            restored_world = deserialize_world_state(payload["snapshots"][-1]["world"])
            self.assertEqual(restored_world.turn, result.world.turn)
            self.assertEqual(restored_world.agents["agent_a"].position, result.world.agents["agent_a"].position)

    def test_default_config_uses_more_competitive_observation_and_smaller_map(self) -> None:
        config = default_config()

        self.assertEqual(config.width, 6)
        self.assertEqual(config.height, 6)
        self.assertEqual(config.observation_radius, 2)
        self.assertFalse(config.competitive_resource_layout)
        self.assertEqual(config.forage_nodes, 8)
        self.assertEqual(config.gold_nodes, 4)

        resources = build_initial_resources(config)
        gold_count = sum(1 for resource in resources.values() if resource.kind is ResourceType.GOLD)
        food_count = sum(1 for resource in resources.values() if resource.kind is ResourceType.FOOD)

        self.assertEqual(gold_count, config.gold_nodes)
        self.assertGreater(food_count, 0)

    def test_scheduler_invokes_turn_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=4, artifact_dir=Path(temp_dir))
            world, agents = build_scripted_simulation(config)
            scheduler = Scheduler(config)
            turns: list[int] = []

            scheduler.run(
                world,
                agents,
                replay_name="callback_run.json",
                on_turn=lambda current_world, _event_bus, _actions: turns.append(current_world.turn),
            )

            self.assertEqual(turns, [1, 2, 3, 4])

    def test_scheduler_stops_when_one_agent_remains_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=10, artifact_dir=Path(temp_dir))
            scheduler = Scheduler(config)
            world = WorldState(
                turn=0,
                width=5,
                height=5,
                agents={
                    "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=1),
                    "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=5),
                },
                resources={},
            )
            actions = {
                "agent_a": ScriptedSurvivor("agent_a", "alpha", TraitProfile.COOPERATIVE),
                "agent_b": ScriptedSurvivor("agent_b", "beta", TraitProfile.GREEDY),
            }

            result = scheduler.run(world, actions, replay_name="last_survivor_stop.json")

            self.assertEqual(result.world.turn, 1)
            self.assertEqual(sum(1 for agent in result.world.agents.values() if agent.alive), 1)

    def test_scheduler_can_continue_until_max_turns_when_last_survivor_stop_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=4, artifact_dir=Path(temp_dir), stop_when_one_agent_remains=False)
            scheduler = Scheduler(config)
            world = WorldState(
                turn=0,
                width=5,
                height=5,
                agents={
                    "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=1),
                    "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=5),
                },
                resources={},
            )
            actions = {
                "agent_a": ScriptedSurvivor("agent_a", "alpha", TraitProfile.COOPERATIVE),
                "agent_b": ScriptedSurvivor("agent_b", "beta", TraitProfile.GREEDY),
            }

            result = scheduler.run(world, actions, replay_name="max_turns_after_last_survivor.json")

            self.assertEqual(result.world.turn, 4)
            self.assertEqual(sum(1 for agent in result.world.agents.values() if agent.alive), 1)

    def test_scheduler_collects_actions_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            action_log: list[tuple[str, str, float]] = []
            config = SimulationConfig(
                max_turns=1,
                artifact_dir=Path(temp_dir),
                memory_enabled=False,
                trust_enabled=False,
                decision_workers=3,
            )
            world = WorldState(
                turn=0,
                width=5,
                height=5,
                agents={
                    "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5),
                    "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=5),
                    "agent_c": AgentState("agent_c", "gamma", Position(2, 0), energy=5),
                },
                resources={},
            )
            agents = {
                "agent_a": SlowAgent("agent_a", "alpha", TraitProfile.COOPERATIVE, action_log=action_log),
                "agent_b": SlowAgent("agent_b", "beta", TraitProfile.GREEDY, action_log=action_log),
                "agent_c": SlowAgent("agent_c", "gamma", TraitProfile.SILENT, action_log=action_log),
            }

            Scheduler(config).run(world, agents, replay_name="parallel.json")

            starts = [timestamp for _agent_id, phase, timestamp in action_log if phase == "start"]
            ends = [timestamp for _agent_id, phase, timestamp in action_log if phase == "end"]
            self.assertEqual(len(starts), 3)
            self.assertEqual(len(ends), 3)
            self.assertLess(max(starts) - min(starts), 0.08)
            self.assertLess(min(ends), max(starts) + 0.14)

    def test_random_simulation_generates_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=5, artifact_dir=Path(temp_dir), random_seed=11)
            world, agents = build_random_simulation(config)
            scheduler = Scheduler(config)

            result = scheduler.run(world, agents, replay_name="random_run.json")

            self.assertGreater(result.metrics.total_messages, 0)
            self.assertGreater(result.metrics.total_message_cost, 0)

    def test_event_schema_and_structured_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=5, artifact_dir=Path(temp_dir), random_seed=11)
            world, agents = build_random_simulation(config)
            scheduler = Scheduler(config)

            result = scheduler.run(world, agents, replay_name="schema_run.json")
            payload = json.loads(result.replay_path.read_text(encoding="utf-8"))
            event_types = [event["event_type"] for event in payload["events"]]

            self.assertEqual(event_types.count(EventType.TURN_START.value), result.metrics.turn)
            self.assertEqual(event_types.count(EventType.TURN_END.value), result.metrics.turn)

            for event in payload["events"]:
                self.assertIn("turn", event)
                self.assertIn("event_type", event)
                self.assertIn("agent_id", event)
                self.assertIn("family_id", event)
                self.assertIn("payload", event)

            message_event = next(event for event in payload["events"] if event["event_type"] == EventType.MESSAGE.value)
            self.assertIn("message", message_event["payload"])
            self.assertEqual(message_event["payload"]["message"]["sender_id"], message_event["agent_id"])
            self.assertIn("content", message_event["payload"]["message"])

    def test_trait_library_covers_all_actions(self) -> None:
        for trait in TraitProfile:
            trait_config = get_trait_config(trait)
            self.assertEqual(set(trait_config.base_action_bias), set(ActionType))
            self.assertEqual(set(trait_config.low_energy_bias), set(ActionType))
            self.assertEqual(set(trait_config.food_visible_bias), set(ActionType))
            self.assertEqual(set(trait_config.gold_visible_bias), set(ActionType))
            self.assertEqual(set(trait_config.nearby_agent_bias), set(ActionType))

    def test_action_scores_reflect_trait_personality(self) -> None:
        world = WorldState(
            turn=1,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=6),
                "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=6),
            },
            resources={Position(0, 1): ResourceNode(kind=ResourceType.GOLD, amount=1)},
        )
        observation = build_observation(world, "agent_a")

        greedy_scores = get_action_scores(observation, get_trait_config(TraitProfile.GREEDY))
        cooperative_scores = get_action_scores(observation, get_trait_config(TraitProfile.COOPERATIVE))
        silent_scores = get_action_scores(observation, get_trait_config(TraitProfile.SILENT))

        self.assertGreater(greedy_scores[ActionType.FORAGE], greedy_scores[ActionType.MESSAGE])
        self.assertGreater(cooperative_scores[ActionType.MESSAGE], greedy_scores[ActionType.MESSAGE])
        self.assertLess(silent_scores[ActionType.MESSAGE], cooperative_scores[ActionType.MESSAGE])

    def test_heuristic_simulation_runs_and_marks_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=5, artifact_dir=Path(temp_dir), random_seed=9)
            world, agents = build_heuristic_simulation(config)
            scheduler = Scheduler(config)

            result = scheduler.run(world, agents, replay_name="heuristic_run.json")
            payload = json.loads(result.replay_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["metadata"]["mode"], "heuristic")
            self.assertEqual(result.metrics.turn, 5)

    def test_cooperative_gold_message_is_structured(self) -> None:
        agent = HeuristicSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=1,
        )
        agent.policy_bias.action_weights[ActionType.MESSAGE] = 10.0
        world = WorldState(
            turn=1,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=6),
                "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=6),
            },
            resources={Position(0, 1): ResourceNode(kind=ResourceType.GOLD, amount=1)},
        )

        action = agent.choose_action(build_observation(world, "agent_a"))

        self.assertEqual(action.action_type, ActionType.MESSAGE)
        self.assertEqual(action.message_intent, MessageIntent.SHARE_GOLD)
        self.assertEqual(action.message_target, (0, 1))
        self.assertEqual(action.resource_hint, ResourceType.GOLD.value)

    def test_gold_can_be_shared_with_nearby_agents(self) -> None:
        config = SimulationConfig(max_turns=1)
        world = WorldState(
            turn=0,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5),
                "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=5),
            },
            resources={Position(0, 0): ResourceNode(kind=ResourceType.GOLD, amount=1)},
        )
        event_bus = EventBus()
        EnvironmentEngine(config).step(
            world,
            {
                "agent_a": AgentAction(action_type=ActionType.FORAGE, share_with_nearby=True),
                "agent_b": AgentAction(action_type=ActionType.REST),
            },
            event_bus,
        )

        cooperation_events = [event for event in event_bus.events if event.event_type is EventType.COOPERATION]
        self.assertEqual(len(cooperation_events), 1)
        self.assertGreater(world.agents["agent_a"].energy, 5)
        self.assertGreater(world.agents["agent_b"].energy, 5)
        self.assertEqual(cooperation_events[0].payload["kind"], "gold_share")

    def test_scheduler_persists_reflections_in_lineage_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(
                max_turns=3,
                artifact_dir=Path(temp_dir),
                initial_energy=1,
                forage_nodes=0,
                gold_nodes=0,
                mode="heuristic",
            )
            world, agents = build_heuristic_simulation(config)
            scheduler = Scheduler(config)

            scheduler.run(world, agents, replay_name="memory_run.json")

            self.assertGreater(scheduler.lineage_store.count(), 0)

    def test_retrieval_engine_returns_structured_family_specific_memories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(artifact_dir=Path(temp_dir), experiment_id="exp-a", run_group="group-a")
            scheduler = Scheduler(config)
            scheduler.lineage_store.add_reflection(
                MemoryRecord(
                    experiment_id=config.experiment_id,
                    run_group=config.run_group,
                    family_id="alpha",
                    lineage_id=config.lineage_id,
                    generation=0,
                    trait=TraitProfile.COOPERATIVE.value,
                    death_reason="energy_depletion",
                    memory_type="death_reflection",
                    source_run_id="run-1",
                    source_agent_id="agent_a",
                    death_turn=5,
                    situation="turn 3 energy 2 gold_visible 0 food_visible 0 nearby_agents 0",
                    lesson="When energy 2 and no food is visible, avoid broadcasts and conserve energy.",
                    tags=["death", "energy_depletion", "overcommunicated"],
                )
            )
            scheduler.lineage_store.add_reflection(
                MemoryRecord(
                    experiment_id=config.experiment_id,
                    run_group=config.run_group,
                    family_id="beta",
                    lineage_id=config.lineage_id,
                    generation=0,
                    trait=TraitProfile.GREEDY.value,
                    death_reason="energy_depletion",
                    memory_type="death_reflection",
                    source_run_id="run-2",
                    source_agent_id="agent_b",
                    death_turn=5,
                    situation="turn 3 energy 2 gold_visible 0 food_visible 0 nearby_agents 0",
                    lesson="Foreign family memory should not be retrieved.",
                    tags=["death"],
                )
            )
            retrieval_engine = RetrievalEngine(scheduler.lineage_store)
            world = WorldState(
                turn=3,
                width=5,
                height=5,
                agents={"agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=2)},
                resources={},
            )
            observation = build_observation(world, "agent_a")

            memories = retrieval_engine.get_relevant_memories(observation, family_id="alpha", lineage_id=config.lineage_id)

            self.assertTrue(memories)
            self.assertIn("avoid broadcasts", memories[0].lesson)
            self.assertGreater(memories[0].score, 0.0)
            self.assertEqual(memories[0].generation, 0)
            self.assertEqual(memories[0].metadata["trait"], TraitProfile.COOPERATIVE.value)
            self.assertEqual(memories[0].metadata["death_reason"], "energy_depletion")
            self.assertEqual(memories[0].metadata["memory_type"], "death_reflection")
            self.assertFalse(any("Foreign family" in memory.lesson for memory in memories))

    def test_lineage_store_isolates_collections_by_run_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            shared_path = Path(temp_dir)
            config_a = SimulationConfig(artifact_dir=shared_path, experiment_id="exp-a", run_group="group-a")
            config_b = SimulationConfig(artifact_dir=shared_path, experiment_id="exp-a", run_group="group-b")
            scheduler_a = Scheduler(config_a)
            scheduler_b = Scheduler(config_b)

            scheduler_a.lineage_store.add_reflection(
                MemoryRecord(
                    experiment_id=config_a.experiment_id,
                    run_group=config_a.run_group,
                    family_id="alpha",
                    lineage_id=config_a.lineage_id,
                    generation=0,
                    trait=TraitProfile.COOPERATIVE.value,
                    death_reason="energy_depletion",
                    memory_type="death_reflection",
                    source_run_id="run-a",
                    source_agent_id="agent_a",
                    death_turn=2,
                    situation="turn 1 energy 1 gold_visible 0 food_visible 0 nearby_agents 0",
                    lesson="Group A memory.",
                    tags=["death"],
                )
            )

            world = WorldState(
                turn=1,
                width=5,
                height=5,
                agents={"agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=1)},
                resources={},
            )
            observation = build_observation(world, "agent_a")

            memories_a = scheduler_a.retrieval_engine.get_relevant_memories(observation, "alpha", config_a.lineage_id)
            memories_b = scheduler_b.retrieval_engine.get_relevant_memories(observation, "alpha", config_b.lineage_id)

            self.assertEqual(len(memories_a), 1)
            self.assertEqual(memories_a[0].lesson, "Group A memory.")
            self.assertEqual(memories_b, [])

    def test_lineage_store_reuses_same_local_qdrant_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(artifact_dir=Path(temp_dir), experiment_id="exp-a", run_group="group-a")
            scheduler_a = Scheduler(config)
            scheduler_b = Scheduler(config)

            scheduler_a.lineage_store.add_reflection(
                MemoryRecord(
                    experiment_id=config.experiment_id,
                    run_group=config.run_group,
                    family_id="alpha",
                    lineage_id=config.lineage_id,
                    generation=0,
                    trait=TraitProfile.COOPERATIVE.value,
                    death_reason="energy_depletion",
                    memory_type="death_reflection",
                    source_run_id="run-1",
                    source_agent_id="agent_a",
                    death_turn=1,
                    situation="turn 1 energy 1 gold_visible 0 food_visible 0 nearby_agents 0",
                    lesson="Shared client memory.",
                    tags=["death"],
                )
            )
            world = WorldState(
                turn=1,
                width=5,
                height=5,
                agents={"agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=1)},
                resources={},
            )

            memories = scheduler_b.retrieval_engine.get_relevant_memories(
                build_observation(world, "agent_a"),
                family_id="alpha",
                lineage_id=config.lineage_id,
            )

            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].lesson, "Shared client memory.")

    def test_scheduler_keeps_agent_memory_interface_as_lessons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(artifact_dir=Path(temp_dir), max_turns=1)
            scheduler = Scheduler(config)
            scheduler.lineage_store.add_reflection(
                MemoryRecord(
                    experiment_id=config.experiment_id,
                    run_group=config.run_group,
                    family_id="alpha",
                    lineage_id=config.lineage_id,
                    generation=0,
                    trait=TraitProfile.COOPERATIVE.value,
                    death_reason="energy_depletion",
                    memory_type="death_reflection",
                    source_run_id="run-1",
                    source_agent_id="agent_a",
                    death_turn=2,
                    situation="turn 0 energy 12 gold_visible 0 food_visible 0 nearby_agents 0",
                    lesson="Preserve energy before broadcasting.",
                    tags=["death"],
                )
            )
            world, agents = build_heuristic_simulation(config)

            scheduler.run(world, agents, replay_name="compatibility.json")

            self.assertIsInstance(agents["agent_a"].memory_package, MemoryContextPackage)
            self.assertTrue(agents["agent_a"].memory_package.typed_lessons)
            self.assertTrue(all(isinstance(memory, str) for memory in agents["agent_a"].memory_context))

    def test_reflection_records_required_stage_four_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(
                artifact_dir=Path(temp_dir),
                max_turns=3,
                initial_energy=1,
                forage_nodes=0,
                gold_nodes=0,
                experiment_id="exp-z",
                run_group="run-group-z",
            )
            world, agents = build_heuristic_simulation(config)
            scheduler = Scheduler(config)

            scheduler.run(world, agents, replay_name="metadata.json")

            memories = scheduler.lineage_store.retrieve(
                family_id="alpha",
                lineage_id=config.lineage_id,
                query_text="energy_depletion low energy no resources",
                memory_type="death_reflection",
            )

            self.assertTrue(memories)
            metadata = memories[0].metadata
            self.assertEqual(metadata["experiment_id"], "exp-z")
            self.assertEqual(metadata["run_group"], "run-group-z")
            self.assertIn(metadata["trait"], {trait.value for trait in TraitProfile})
            self.assertEqual(metadata["memory_type"], "death_reflection")
            self.assertEqual(metadata["death_reason"], "energy_depletion")

    def test_memory_context_is_injected_into_heuristic_agent(self) -> None:
        agent = HeuristicSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=3,
        )
        agent.set_memory_context(["Reduce broadcasts when low on energy; communication cost can outweigh survival gains."])

        world = WorldState(
            turn=2,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=2),
                "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=5),
            },
            resources={},
        )
        observation = build_observation(world, "agent_a")
        base_scores = get_action_scores(observation, get_trait_config(TraitProfile.COOPERATIVE))
        agent.choose_action(observation)

        self.assertLess(agent.policy_bias.action_weights[ActionType.MESSAGE], 1.0)
        self.assertGreater(base_scores[ActionType.MESSAGE], -1.0)

    def test_memory_package_blocks_suspicious_reroute_for_heuristic_agent(self) -> None:
        agent = HeuristicSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=7,
            temperature=0.01,
        )
        agent.policy_bias.action_weights[ActionType.MOVE] = 8.0
        agent.set_memory_context(
            MemoryContextPackage(
                hard_constraints=["Do not reroute solely on unverified suspicious signals."],
                typed_lessons=["[deception] Ignore suspicious false gold signals until validated."],
                directives=[
                    MemoryDirective(
                        memory_type="deception_reflection",
                        priority=0.9,
                        lesson="Ignore suspicious false gold signals until validated.",
                        hard_constraint="Do not reroute solely on unverified suspicious signals.",
                        action_bias={"move": 0.2},
                        target_preference="seek_verified_resource",
                        caution_against="false_gold",
                    )
                ],
            )
        )
        world = WorldState(
            turn=3,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5),
                "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=5),
            },
            resources={},
        )
        observation = build_observation(
            world,
            "agent_a",
            social_hints=[
                SocialHint(
                    sender_id="agent_b",
                    sender_family_id="beta",
                    intent=MessageIntent.FALSE_GOLD.value,
                    resource_hint=ResourceType.GOLD.value,
                    target_position=Position(2, 0),
                    trust_score=-0.6,
                    sender_reputation=-0.5,
                    message_utility=-0.7,
                    alliance_likelihood=-0.6,
                    threat_level=1.1,
                )
            ],
            recent_received_messages=[
                ReceivedMessageSummary(
                    turn=2,
                    sender_id="agent_b",
                    sender_family_id="beta",
                    content="gold east",
                    intent=MessageIntent.FALSE_GOLD.value,
                    resource_hint=ResourceType.GOLD.value,
                    target_position=Position(2, 0),
                    trust_score=-0.6,
                    sender_reputation=-0.5,
                    message_utility=-0.7,
                    alliance_likelihood=-0.6,
                    threat_level=1.1,
                )
            ],
            nearby_unexplored_positions=[Position(0, 1)],
        )

        action = agent.choose_action(observation)

        self.assertEqual(action.action_type, ActionType.MOVE)
        self.assertEqual(action.direction, Direction.DOWN)

    def test_heuristic_agent_breaks_simple_backtrack_loop(self) -> None:
        agent = HeuristicSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=5,
        )
        agent.policy_bias.action_weights[ActionType.MOVE] = 8.0

        world = WorldState(
            turn=6,
            width=5,
            height=5,
            agents={"agent_a": AgentState("agent_a", "alpha", Position(1, 1), energy=6)},
            resources={},
        )
        observation = build_observation(
            world,
            "agent_a",
            recent_positions=[Position(1, 1), Position(2, 1), Position(1, 1), Position(2, 1)],
            recent_self_events=[
                RecentEventSummary(turn=3, event_type="move", position=Position(1, 1), detail="left"),
                RecentEventSummary(turn=4, event_type="move", position=Position(2, 1), detail="right"),
            ],
            unique_positions_visited=2,
        )

        action = agent.choose_action(observation)

        self.assertEqual(action.action_type, ActionType.MOVE)
        self.assertIn(action.direction, {Direction.UP, Direction.DOWN})

    def test_heuristic_agent_moves_toward_resource_hotspot(self) -> None:
        agent = HeuristicSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=4,
            temperature=0.01,
        )
        agent.policy_bias.action_weights[ActionType.MOVE] = 8.0
        world = WorldState(
            turn=4,
            width=5,
            height=5,
            agents={"agent_a": AgentState("agent_a", "alpha", Position(1, 1), energy=6)},
            resources={},
        )

        observation = build_observation(
            world,
            "agent_a",
            resource_hotspots=[
                ResourceHotspot(
                    position=Position(3, 1),
                    resource_kind=ResourceType.GOLD.value,
                    sightings=3,
                    last_seen_turn=3,
                )
            ],
        )

        action = agent.choose_action(observation)

        self.assertEqual(action.action_type, ActionType.MOVE)
        self.assertEqual(action.direction, Direction.RIGHT)

    def test_scheduler_prepares_extended_observation_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=2, artifact_dir=Path(temp_dir), trust_enabled=True)
            scheduler = Scheduler(config)
            world = WorldState(
                turn=1,
                width=5,
                height=5,
                agents={
                    "agent_a": AgentState("agent_a", "alpha", Position(1, 1), energy=5),
                    "agent_b": AgentState("agent_b", "beta", Position(2, 1), energy=5),
                },
                resources={Position(1, 2): ResourceNode(kind=ResourceType.GOLD, amount=1)},
            )
            agents = {
                "agent_a": HeuristicSurvivor("agent_a", "alpha", TraitProfile.COOPERATIVE),
                "agent_b": HeuristicSurvivor("agent_b", "beta", TraitProfile.GREEDY),
            }
            event_bus = EventBus()
            event_bus.record(
                turn=1,
                event_type=EventType.MESSAGE,
                agent_id="agent_b",
                family_id="beta",
                payload={
                    "message": {
                        "sender_id": "agent_b",
                        "sender_family_id": "beta",
                        "content": "bx",
                        "intent": MessageIntent.FALSE_GOLD.value,
                        "target_position": {"x": 1, "y": 2},
                        "resource_hint": ResourceType.GOLD.value,
                    },
                    "cost": 2,
                },
            )
            scheduler.trust_tracker.process_events(event_bus, world)
            scheduler.trust_tracker.scores = {"agent_a": {"agent_b": -0.6}}
            position_history = {
                "agent_a": [Position(0, 1), Position(1, 1)],
                "agent_b": [Position(2, 1)],
            }
            resource_history = {"agent_a": {}, "agent_b": {}}

            pending = scheduler._prepare_pending_decisions(world, agents, event_bus, position_history, resource_history)
            observation = next(decision.observation for decision in pending if decision.agent_id == "agent_a")

            self.assertEqual(len(observation.recent_received_messages), 1)
            self.assertEqual(observation.recent_received_messages[0].sender_id, "agent_b")
            self.assertTrue(observation.nearby_unexplored_positions)
            self.assertAlmostEqual(observation.exploration_ratio, 0.08)
            self.assertTrue(observation.resource_hotspots)
            received = observation.recent_received_messages[0]
            self.assertLess(received.message_utility, 0.0)
            self.assertGreater(received.threat_level, 0.5)
            profile = next(profile for profile in observation.agent_profiles if profile.agent_id == "agent_b")
            self.assertEqual(profile.message_count, 1)
            self.assertEqual(profile.false_gold_count, 1)
            self.assertLess(profile.truth_score, 0.0)
            self.assertLess(profile.alliance_likelihood, 0.0)
            self.assertGreater(profile.threat_level, 0.0)

    def test_scheduler_omits_social_state_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=2, artifact_dir=Path(temp_dir), trust_enabled=False, social_reasoning_enabled=False)
            scheduler = Scheduler(config)
            world = WorldState(
                turn=1,
                width=5,
                height=5,
                agents={
                    "agent_a": AgentState("agent_a", "alpha", Position(1, 1), energy=5),
                    "agent_b": AgentState("agent_b", "beta", Position(2, 1), energy=5),
                },
                resources={},
            )
            agents = {
                "agent_a": HeuristicSurvivor("agent_a", "alpha", TraitProfile.COOPERATIVE),
                "agent_b": HeuristicSurvivor("agent_b", "beta", TraitProfile.GREEDY),
            }

            pending = scheduler._prepare_pending_decisions(world, agents, EventBus(), {"agent_a": [], "agent_b": []}, {"agent_a": {}, "agent_b": {}})
            observation = next(decision.observation for decision in pending if decision.agent_id == "agent_a")

            self.assertEqual(observation.social_hints, [])
            self.assertEqual(observation.recent_received_messages, [])
            self.assertEqual(observation.agent_profiles, [])

    def test_llm_prompt_includes_extended_observation_context(self) -> None:
        world = WorldState(
            turn=5,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(1, 1), energy=4),
                "agent_b": AgentState("agent_b", "beta", Position(2, 1), energy=6),
            },
            resources={},
        )
        observation = build_observation(
            world,
            "agent_a",
            recent_received_messages=[
                ReceivedMessageSummary(
                    turn=4,
                    sender_id="agent_b",
                    sender_family_id="beta",
                    content="gold east",
                    intent=MessageIntent.SHARE_GOLD.value,
                    resource_hint=ResourceType.GOLD.value,
                    target_position=Position(2, 1),
                    trust_score=0.7,
                )
            ],
            explored_positions=[Position(0, 1), Position(1, 1)],
            nearby_unexplored_positions=[Position(2, 1), Position(2, 2)],
            resource_hotspots=[
                ResourceHotspot(Position(3, 1), ResourceType.GOLD.value, sightings=2, last_seen_turn=4)
            ],
            agent_profiles=[
                AgentSocialProfile(
                    agent_id="agent_b",
                    family_id="beta",
                    truth_score=0.7,
                    message_count=3,
                    false_gold_count=0,
                    gold_competition_count=2,
                )
            ],
            unique_positions_visited=2,
            exploration_ratio=0.08,
        )

        _system_prompt, user_prompt = build_llm_prompts(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE.value,
            observation=observation,
            memory_context=MemoryContextPackage(
                hard_constraints=["Do not overreact to suspicious signals without nearby evidence."],
                soft_hints=["Trust consistent gold signals more than isolated broadcasts."],
                examples=["When trusted hints matched visible gold, moving toward them produced gains."],
                typed_lessons=["[deception] Cross-check suspicious claims before rerouting."],
            ),
            current_plan=ShortTermPlan(current_goal="reach trusted gold", planned_target_position=(3, 1), created_turn=3),
            policy_bias={action_type.value: 0.0 for action_type in ActionType},
            heuristic_recommendation=AgentAction(
                action_type=ActionType.MOVE,
                direction=Direction.RIGHT,
                current_goal="reach trusted gold",
                planned_target_position=(3, 1),
            ),
        )
        payload = json.loads(user_prompt)

        self.assertIn("recent_received_messages", payload["observation"]["history"])
        self.assertIn("resource_hotspots", payload["observation"]["history"])
        self.assertEqual(payload["observation"]["history"]["recent_received_messages"][0]["content"], "gold east")
        self.assertEqual(payload["observation"]["agent_profiles"][0]["message_count"], 3)
        self.assertEqual(payload["planning"]["current_plan"]["goal"], "reach trusted gold")
        self.assertIn("hard_constraints", payload["family_memory"])
        self.assertIn("soft_hints", payload["family_memory"])
        self.assertIn("examples", payload["family_memory"])
        self.assertIn("neural_symbolic_fusion", payload)
        self.assertIn("Fast instinct layer ranks actions", payload["neural_symbolic_fusion"]["instinct_summary"])
        self.assertIn("Retrieved family memory suggests", payload["neural_symbolic_fusion"]["memory_summary"])
        self.assertIn("Heuristic draft action", payload["neural_symbolic_fusion"]["heuristic_recommendation"])

    def test_scheduler_stores_success_and_cooperation_memories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(
                max_turns=1,
                artifact_dir=Path(temp_dir),
                mode="heuristic",
            )
            world = WorldState(
                turn=0,
                width=5,
                height=5,
                agents={
                    "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5),
                    "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=5),
                    "agent_c": AgentState("agent_c", "gamma", Position(4, 4), energy=5),
                },
                resources={Position(0, 0): ResourceNode(kind=ResourceType.GOLD, amount=1)},
            )
            agents = {
                "agent_a": HeuristicSurvivor("agent_a", "alpha", TraitProfile.COOPERATIVE),
                "agent_b": HeuristicSurvivor("agent_b", "beta", TraitProfile.COOPERATIVE),
                "agent_c": HeuristicSurvivor("agent_c", "gamma", TraitProfile.SILENT),
            }
            agents["agent_a"].policy_bias.action_weights[ActionType.FORAGE] = 10.0
            scheduler = Scheduler(config)

            scheduler.run(world, agents, replay_name="memory_types.json")

            cooperation_memories = scheduler.lineage_store.retrieve(
                family_id="alpha",
                lineage_id=config.lineage_id,
                query_text="cooperate gold share nearby allies",
                memory_type="cooperation_reflection",
            )
            success_memories = scheduler.lineage_store.retrieve(
                family_id="gamma",
                lineage_id=config.lineage_id,
                query_text="survive stable energy resourceful",
                memory_type="success_reflection",
            )

            self.assertTrue(cooperation_memories)
            self.assertTrue(success_memories)

    def test_reflection_engine_creates_deception_memories(self) -> None:
        event_bus = EventBus()
        world = WorldState(
            turn=2,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5),
                "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=5),
            },
            resources={},
        )
        event_bus.record(
            turn=1,
            event_type=EventType.MESSAGE,
            agent_id="agent_b",
            family_id="beta",
            payload={
                "message": {
                    "sender_id": "agent_b",
                    "sender_family_id": "beta",
                    "content": "bx",
                    "intent": MessageIntent.FALSE_GOLD.value,
                    "target_position": {"x": 3, "y": 3},
                    "resource_hint": ResourceType.GOLD.value,
                },
                "cost": 2,
            },
        )
        event_bus.record(
            turn=2,
            event_type=EventType.TRUST_UPDATE,
            agent_id="agent_a",
            family_id="alpha",
            payload={
                "sender_id": "agent_b",
                "delta": -1.3,
                "score": -1.3,
                "reason": "misleading_hint",
            },
        )

        reflections = ReflectionEngine().summarize_run_memories(
            event_bus,
            world,
            experiment_id="exp-d",
            run_group="group-d",
            lineage_id="mixed_population",
            generation=0,
            run_id="run-d",
            reflection_window=6,
            agent_traits={"agent_a": TraitProfile.COOPERATIVE.value, "agent_b": TraitProfile.GREEDY.value},
        )

        deception_reflections = [reflection for reflection in reflections if reflection.memory_type == "deception_reflection"]
        self.assertEqual(len(deception_reflections), 1)
        self.assertIn("suspicious", deception_reflections[0].lesson)
        self.assertIn("false_gold", deception_reflections[0].tags)

    def test_retrieval_engine_prioritizes_deception_memories_for_suspicious_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(artifact_dir=Path(temp_dir), experiment_id="exp-r", run_group="group-r")
            scheduler = Scheduler(config)
            scheduler.lineage_store.add_reflection(
                MemoryRecord(
                    experiment_id=config.experiment_id,
                    run_group=config.run_group,
                    family_id="alpha",
                    lineage_id=config.lineage_id,
                    generation=0,
                    trait=TraitProfile.COOPERATIVE.value,
                    death_reason="misleading_hint",
                    memory_type="deception_reflection",
                    source_run_id="run-1",
                    source_agent_id="agent_a",
                    death_turn=4,
                    situation="receiver agent_a sender agent_b intent false_gold",
                    lesson="Treat repeated false gold claims as suspicious until locally verified.",
                    tags=["deception", "false_gold"],
                )
            )
            scheduler.lineage_store.add_reflection(
                MemoryRecord(
                    experiment_id=config.experiment_id,
                    run_group=config.run_group,
                    family_id="alpha",
                    lineage_id=config.lineage_id,
                    generation=0,
                    trait=TraitProfile.COOPERATIVE.value,
                    death_reason="survived",
                    memory_type="success_reflection",
                    source_run_id="run-2",
                    source_agent_id="agent_a",
                    death_turn=6,
                    situation="survived with stable energy",
                    lesson="Maintain balanced exploration when no suspicious signals exist.",
                    tags=["survival"],
                )
            )
            observation = build_observation(
                WorldState(
                    turn=3,
                    width=5,
                    height=5,
                    agents={"agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5)},
                    resources={},
                ),
                "agent_a",
                recent_received_messages=[
                    ReceivedMessageSummary(
                        turn=2,
                        sender_id="agent_b",
                        sender_family_id="beta",
                        content="gold there",
                        intent=MessageIntent.FALSE_GOLD.value,
                        resource_hint=ResourceType.GOLD.value,
                        target_position=Position(3, 3),
                        trust_score=-0.4,
                        sender_reputation=-0.3,
                        message_utility=-0.5,
                        alliance_likelihood=-0.4,
                        threat_level=0.9,
                    )
                ],
            )

            memories = scheduler.retrieval_engine.get_relevant_memories(observation, family_id="alpha", lineage_id=config.lineage_id)
            lessons = scheduler.retrieval_engine.get_memory_lessons(observation, family_id="alpha", lineage_id=config.lineage_id)

            self.assertTrue(memories)
            self.assertEqual(memories[0].metadata["memory_type"], "deception_reflection")
            self.assertTrue(lessons[0].startswith("[deception]"))

    def test_greedy_agent_can_emit_false_gold_signal(self) -> None:
        agent = HeuristicSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.GREEDY,
            seed=2,
        )
        agent.policy_bias.action_weights[ActionType.MESSAGE] = 12.0
        world = WorldState(
            turn=1,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=6),
                "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=6),
            },
            resources={Position(0, 1): ResourceNode(kind=ResourceType.FOOD, amount=1)},
        )

        action = agent.choose_action(build_observation(world, "agent_a"))

        self.assertEqual(action.action_type, ActionType.MESSAGE)
        self.assertEqual(action.message_intent, MessageIntent.FALSE_GOLD)
        self.assertEqual(action.resource_hint, ResourceType.GOLD.value)
        self.assertEqual(action.message_target, (0, 1))

    def test_trust_tracker_penalizes_misleading_gold_signal(self) -> None:
        config = SimulationConfig(trust_window=3)
        tracker = TrustTracker(config)
        world = WorldState(
            turn=0,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=6),
                "agent_b": AgentState("agent_b", "beta", Position(1, 0), energy=6),
            },
            resources={},
        )
        event_bus = EventBus()
        event_bus.record(
            turn=1,
            event_type=EventType.MESSAGE,
            agent_id="agent_a",
            family_id="alpha",
            payload={
                "message": {
                    "sender_id": "agent_a",
                    "sender_family_id": "alpha",
                    "content": "gx",
                    "content_length": 2,
                    "channel": "broadcast",
                    "intent": MessageIntent.FALSE_GOLD.value,
                    "target_position": {"x": 0, "y": 0},
                    "resource_hint": ResourceType.GOLD.value,
                },
                "cost": 2,
            },
        )
        tracker.process_events(event_bus, world)
        event_bus.record(
            turn=2,
            event_type=EventType.FORAGE_MISS,
            agent_id="agent_b",
            family_id="beta",
            payload={"position": {"x": 0, "y": 0}},
        )
        world.turn = 2

        tracker.process_events(event_bus, world)

        self.assertLess(tracker.get_score("agent_b", "agent_a"), 0.0)
        assessment = tracker.assess_signal("agent_b", "beta", "agent_a", "alpha", MessageIntent.FALSE_GOLD.value)
        self.assertLess(assessment.message_utility, 0.0)
        self.assertGreater(assessment.threat_level, 0.5)
        trust_updates = [event for event in event_bus.events if event.event_type is EventType.TRUST_UPDATE]
        self.assertEqual(len(trust_updates), 1)
        self.assertEqual(trust_updates[0].payload["reason"], "misleading_hint")

    def test_heuristic_agent_moves_toward_trusted_hint(self) -> None:
        agent = HeuristicSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=5,
            temperature=0.01,
        )
        agent.policy_bias.action_weights[ActionType.MOVE] = 8.0
        world = WorldState(
            turn=2,
            width=5,
            height=5,
            agents={
                "agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5),
                "agent_b": AgentState("agent_b", "beta", Position(2, 0), energy=5),
            },
            resources={},
        )
        observation = build_observation(
            world,
            "agent_a",
            social_hints=[
                SocialHint(
                    sender_id="agent_b",
                    sender_family_id="beta",
                    intent=MessageIntent.SHARE_GOLD.value,
                    resource_hint=ResourceType.GOLD.value,
                    target_position=Position(2, 0),
                    trust_score=0.8,
                    sender_reputation=0.7,
                    message_utility=0.8,
                    alliance_likelihood=0.75,
                    threat_level=0.1,
                )
            ],
        )

        action = agent.choose_action(observation)

        self.assertEqual(action.action_type, ActionType.MOVE)
        self.assertEqual(action.direction.value, "right")

    def test_cognition_graph_retries_then_accepts_valid_json(self) -> None:
        graph = CognitionGraph(FakeLLMAdapter(["oops", '{"action_type":"rest"}']))
        fallback = AgentAction(action_type=ActionType.MOVE, direction=None)
        world = WorldState(
            turn=1,
            width=5,
            height=5,
            agents={"agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5)},
            resources={},
        )

        action = graph.run(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE.value,
            observation=build_observation(world, "agent_a"),
            memory_context=["Preserve energy."],
            current_plan=None,
            policy_bias={action_type.value: 0.0 for action_type in ActionType},
            fallback_action=fallback,
        )

        self.assertEqual(action.action_type, ActionType.REST)
        self.assertEqual(action.decision_source, "llm_repair")

    def test_cognition_graph_falls_back_after_invalid_model_output(self) -> None:
        fallback = AgentAction(action_type=ActionType.REST)
        graph = CognitionGraph(FakeLLMAdapter(["@@@", "{bad json"]), max_retries=1)
        world = WorldState(
            turn=1,
            width=5,
            height=5,
            agents={"agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5)},
            resources={},
        )

        action = graph.run(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE.value,
            observation=build_observation(world, "agent_a"),
            memory_context=[],
            current_plan=None,
            policy_bias={action_type.value: 0.0 for action_type in ActionType},
            fallback_action=fallback,
        )

        self.assertEqual(action.action_type, fallback.action_type)
        self.assertEqual(action.decision_source, "heuristic_fallback")

    def test_llm_agent_uses_structured_action_output(self) -> None:
        adapter = FakeLLMAdapter([
            '{"action_type":"move","direction":"right","current_goal":"reach eastern hotspot","planned_target_position":{"x":2,"y":0}}'
        ])
        agent = LLMSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            llm_adapter=adapter,
        )
        agent.set_memory_context(
            MemoryContextPackage(
                hard_constraints=["Do not spend energy chasing suspicious gold without nearby evidence."],
                soft_hints=["A visible eastern path is usually safer than a blind detour."],
            )
        )
        world = WorldState(
            turn=1,
            width=5,
            height=5,
            agents={"agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5)},
            resources={},
        )

        action = agent.choose_action(build_observation(world, "agent_a"))

        self.assertEqual(action.action_type, ActionType.MOVE)
        self.assertEqual(action.direction.value, "right")
        self.assertEqual(action.decision_source, "llm")
        self.assertEqual(agent.current_plan.current_goal, "reach eastern hotspot")
        self.assertEqual(agent.current_plan.planned_target_position, (2, 0))
        self.assertTrue(adapter.calls)
        payload = json.loads(adapter.calls[-1][1])
        self.assertIn("neural_symbolic_fusion", payload)
        self.assertIn("Fast instinct layer ranks actions", payload["neural_symbolic_fusion"]["instinct_summary"])
        self.assertIn("Retrieved family memory suggests", payload["neural_symbolic_fusion"]["memory_summary"])
        self.assertIn("Heuristic draft action", payload["neural_symbolic_fusion"]["heuristic_recommendation"])

    def test_llm_agent_fallback_preserves_active_plan(self) -> None:
        agent = LLMSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=1,
            temperature=0.01,
            llm_adapter=FakeLLMAdapter([
                '{"action_type":"move","direction":"right","current_goal":"reach eastern hotspot","planned_target_position":{"x":2,"y":0}}',
                '@@@',
                '@@@',
            ]),
        )
        first_world = WorldState(
            turn=1,
            width=5,
            height=5,
            agents={"agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5)},
            resources={},
        )
        first_action = agent.choose_action(build_observation(first_world, "agent_a"))

        second_world = WorldState(
            turn=2,
            width=5,
            height=5,
            agents={"agent_a": AgentState("agent_a", "alpha", Position(1, 0), energy=5)},
            resources={},
        )
        second_action = agent.choose_action(build_observation(second_world, "agent_a"))

        self.assertEqual(first_action.planned_target_position, (2, 0))
        self.assertEqual(second_action.decision_source, "heuristic_fallback")
        self.assertEqual(second_action.planned_target_position, (2, 0))
        self.assertEqual(second_action.direction, Direction.RIGHT)

    def test_llm_agent_can_disable_planning(self) -> None:
        agent = LLMSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            planning_enabled=False,
            llm_adapter=FakeLLMAdapter([
                '{"action_type":"move","direction":"right","current_goal":"reach eastern hotspot","planned_target_position":{"x":2,"y":0}}'
            ]),
        )
        world = WorldState(
            turn=1,
            width=5,
            height=5,
            agents={"agent_a": AgentState("agent_a", "alpha", Position(0, 0), energy=5)},
            resources={},
        )

        action = agent.choose_action(build_observation(world, "agent_a"))

        self.assertEqual(action.planned_target_position, (2, 0))
        self.assertTrue(agent.current_plan.is_empty())

    def test_llm_simulation_runs_with_scripted_agents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(max_turns=4, artifact_dir=Path(temp_dir), random_seed=13)
            responses = {
                "agent_a": FakeLLMAdapter(['{"action_type":"move","direction":"right"}'] * 4),
                "agent_b": FakeLLMAdapter(['{"action_type":"rest"}'] * 4),
                "agent_c": FakeLLMAdapter(['{"action_type":"move","direction":"down"}'] * 4),
            }
            world, agents = build_llm_simulation(config, llm_adapter_factory=lambda agent_id: responses[agent_id])
            scheduler = Scheduler(config)

            result = scheduler.run(world, agents, replay_name="llm_run.json")
            payload = json.loads(result.replay_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["metadata"]["mode"], "llm")
            self.assertEqual(result.metrics.turn, 4)
            self.assertIn("agent_a", payload["agents"])
            self.assertTrue(all(isinstance(agent, LLMSurvivor) for agent in agents.values()))

    def test_generational_experiment_archives_manifest_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
            config_a = SimulationConfig(
                max_turns=4,
                artifact_dir=Path(temp_dir_a),
                experiment_id="exp-archive",
                run_group="group-archive",
                random_seed=19,
            )
            config_b = SimulationConfig(
                max_turns=4,
                artifact_dir=Path(temp_dir_b),
                experiment_id="exp-archive",
                run_group="group-archive",
                random_seed=19,
            )

            summary_a = run_generational_experiment(
                config_a,
                generations=2,
                runs_per_generation=2,
                ablation_mode="no_memory",
            )
            summary_b = run_generational_experiment(
                config_b,
                generations=2,
                runs_per_generation=2,
                ablation_mode="no_memory",
            )

            manifest_a = json.loads(Path(summary_a["manifest_path"]).read_text(encoding="utf-8"))
            manifest_b = json.loads(Path(summary_b["manifest_path"]).read_text(encoding="utf-8"))

            self.assertEqual(summary_a["run_count"], 4)
            self.assertEqual(len(manifest_a["generation_summaries"]), 2)
            self.assertEqual(
                [row["run_id"] for row in manifest_a["run_summaries"]],
                [row["run_id"] for row in manifest_b["run_summaries"]],
            )
            for run_record in manifest_a["runs"]:
                replay_path = Path(run_record["replay_path"])
                self.assertTrue(replay_path.exists())
                self.assertIn(f"generation_{run_record['generation']:03d}", str(replay_path))

    def test_experiment_catalog_discovers_generation_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)
            manifest_dir = artifact_root / "exp-a" / "group-a"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / "experiment_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "experiment": {
                            "experiment_id": "exp-a",
                            "run_group": "group-a",
                            "lineage_id": "mixed_population",
                            "ablation_mode": "llm_with_memory",
                            "generations": 3,
                            "runs_per_generation": 2,
                        },
                        "generation_summaries": [
                            {"generation": 0, "average_survival_turn": 3.0, "entropy": 1.2, "deception_frequency": 0.1},
                            {"generation": 1, "average_survival_turn": 4.5, "entropy": 1.0, "deception_frequency": 0.2},
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            catalog = build_experiment_catalog(artifact_root)
            survival_rows = build_generation_metric_rows(catalog[0]["manifest"], "average_survival_turn")

            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0]["experiment_id"], "exp-a")
            self.assertEqual(catalog[0]["ablation_mode"], "llm_with_memory")
            self.assertEqual(catalog[0]["latest_summary"]["average_survival_turn"], 4.5)
            self.assertEqual(survival_rows, [
                {"generation": 0, "average_survival_turn": 3.0},
                {"generation": 1, "average_survival_turn": 4.5},
            ])

    def test_baseline_benchmark_produces_group_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(
                max_turns=3,
                artifact_dir=Path(temp_dir),
                experiment_id="exp-benchmark",
                run_group="group-benchmark",
                random_seed=23,
            )

            summary = run_baseline_benchmark(
                config,
                runs_per_group=2,
                llm_adapter_factory=lambda _agent_id: FakeLLMAdapter(['{"action_type":"rest"}'] * 12),
            )

            manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
            groups = {row["benchmark_group"]: row for row in manifest["group_summaries"]}

            self.assertEqual(summary["run_count"], 10)
            self.assertEqual(set(groups), {"scripted", "heuristic", "llm", "llm_without_memory", "llm_with_memory"})
            self.assertFalse(groups["scripted"]["memory_enabled"])
            self.assertFalse(groups["heuristic"]["memory_enabled"])
            self.assertFalse(groups["llm_without_memory"]["memory_enabled"])
            self.assertTrue(groups["llm_with_memory"]["memory_enabled"])
            self.assertIn("average_survival_turn", groups["llm"])
            self.assertIn("resource_acquisition_rate", groups["llm"])
            self.assertIn("message_cost_per_turn", groups["llm"])
            self.assertIn("cooperation_rate", groups["llm"])
            self.assertIn("deception_frequency", groups["llm"])
            self.assertEqual(len(manifest["runs"]), 10)
            self.assertTrue(all(Path(run["replay_path"]).exists() for run in manifest["runs"]))

    def test_extended_benchmark_captures_planning_and_social_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(
                max_turns=2,
                artifact_dir=Path(temp_dir),
                experiment_id="exp-extended-benchmark",
                run_group="group-extended-benchmark",
                random_seed=31,
            )

            summary = run_baseline_benchmark(
                config,
                runs_per_group=1,
                benchmark_groups=("llm", "llm_without_planning", "llm_without_social_reasoning"),
                llm_adapter_factory=lambda _agent_id: FakeLLMAdapter(['{"action_type":"rest"}'] * 6),
            )

            manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
            groups = {row["benchmark_group"]: row for row in manifest["group_summaries"]}

            self.assertEqual(set(groups), {"llm", "llm_without_planning", "llm_without_social_reasoning"})
            self.assertTrue(groups["llm"]["planning_enabled"])
            self.assertFalse(groups["llm_without_planning"]["planning_enabled"])
            self.assertTrue(groups["llm"]["social_reasoning_enabled"])
            self.assertFalse(groups["llm_without_social_reasoning"]["social_reasoning_enabled"])

    def test_communication_analysis_reports_stage_seven_statistics(self) -> None:
        event_bus = EventBus()
        event_bus.record(
            turn=1,
            event_type=EventType.MESSAGE,
            agent_id="agent_a",
            family_id="alpha",
            payload={
                "message": {
                    "content": "food food gold",
                    "intent": MessageIntent.SHARE_GOLD.value,
                },
                "cost": 3,
            },
        )
        event_bus.record(
            turn=2,
            event_type=EventType.MESSAGE,
            agent_id="agent_b",
            family_id="beta",
            payload={
                "message": {
                    "content": "fake gold",
                    "intent": MessageIntent.FALSE_GOLD.value,
                },
                "cost": 2,
            },
        )

        report = CommunicationAnalysis().analyze(event_bus)

        self.assertIn("food", report.word_frequency)
        self.assertGreater(report.protocol_compression_rate, 0.0)
        self.assertAlmostEqual(report.deception_frequency, 0.5)
        self.assertEqual(report.false_gold_signals, 1)


if __name__ == "__main__":
    unittest.main()