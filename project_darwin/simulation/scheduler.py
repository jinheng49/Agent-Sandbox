from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from project_darwin.agents.action_space import AgentAction
from project_darwin.agents.base_agent import BaseAgent
from project_darwin.analytics.communication_analysis import CommunicationAnalysis, CommunicationReport
from project_darwin.analytics.metrics_engine import MetricsEngine, MetricsSnapshot, RunSummary
from project_darwin.analytics.replay_store import ReplayStore
from project_darwin.environment.env_engine import EnvironmentEngine
from project_darwin.environment.observation_builder import (
    AgentSocialProfile,
    Observation,
    ReceivedMessageSummary,
    RecentEventSummary,
    ResourceHotspot,
    build_observation,
)
from project_darwin.memory.lineage_store import LineageStore
from project_darwin.memory.retrieval_engine import RetrievalEngine
from project_darwin.memory.reflection_engine import ReflectionEngine
from project_darwin.simulation.event_bus import EventBus, EventType
from project_darwin.simulation.run_context import SimulationConfig
from project_darwin.simulation.state import Position, WorldState, serialize_world_state
from project_darwin.simulation.trust_tracker import TrustTracker


TurnCallback = Callable[[WorldState, EventBus, dict[str, AgentAction]], None]


@dataclass(slots=True)
class PendingDecision:
    agent_id: str
    observation: Observation


@dataclass(slots=True)
class SimulationResult:
    world: WorldState
    metrics: MetricsSnapshot
    communication: CommunicationReport
    run_summary: RunSummary
    metadata: dict[str, object]
    replay_path: Path


class Scheduler:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.environment = EnvironmentEngine(config)
        self.metrics_engine = MetricsEngine()
        self.communication_analysis = CommunicationAnalysis()
        self.replay_store = ReplayStore()
        self.reflection_engine = ReflectionEngine()
        self.lineage_store = LineageStore(config)
        self.retrieval_engine = RetrievalEngine(self.lineage_store)
        self.trust_tracker = TrustTracker(config)

    def run(
        self,
        world: WorldState,
        agents: dict[str, BaseAgent],
        replay_name: str = "latest_run.json",
        on_turn: TurnCallback | None = None,
        archive_run: bool = False,
    ) -> SimulationResult:
        event_bus = EventBus()
        run_metadata = self._build_run_metadata(world, agents)
        snapshots = [self._build_snapshot(world, event_bus)]
        position_history = {
            agent_id: [agent.position]
            for agent_id, agent in world.agents.items()
        }
        resource_history: dict[str, dict[tuple[int, int, str], dict[str, int]]] = {
            agent_id: {}
            for agent_id in world.agents
        }

        while world.turn < self.config.max_turns and self._should_continue(world):
            current_turn = world.turn + 1
            event_bus.record(
                turn=current_turn,
                event_type=EventType.TURN_START,
                payload={
                    "alive_agents": self._alive_agent_count(world),
                    "resources_remaining": len(world.resources),
                },
            )
            pending_decisions = self._prepare_pending_decisions(world, agents, event_bus, position_history, resource_history)
            actions = self._collect_actions(agents, pending_decisions)

            self.environment.step(world, actions, event_bus)
            self._append_position_history(world, position_history)
            self.trust_tracker.process_events(event_bus, world)
            event_bus.record(
                turn=world.turn,
                event_type=EventType.TURN_END,
                payload={
                    "alive_agents": self._alive_agent_count(world),
                    "resources_remaining": len(world.resources),
                },
            )
            snapshots.append(self._build_snapshot(world, event_bus))
            if on_turn is not None:
                on_turn(world, event_bus, actions)

        reflections = self.reflection_engine.summarize_run_memories(
            event_bus,
            world,
            experiment_id=self.config.experiment_id,
            run_group=self.config.run_group,
            lineage_id=self.config.lineage_id,
            generation=self.config.generation,
            run_id=str(run_metadata["run_id"]),
            reflection_window=self.config.reflection_window,
            agent_traits={agent_id: agent.trait.value for agent_id, agent in agents.items()},
        )
        for reflection in reflections:
            self.lineage_store.add_reflection(reflection)

        metrics = self.metrics_engine.snapshot(world, event_bus)
        communication = self.communication_analysis.analyze(event_bus)
        run_summary = self.metrics_engine.summarize_run(world, event_bus, metrics, communication, run_metadata)
        replay_path = self._build_replay_path(run_metadata, replay_name, archive_run)
        self.replay_store.write_run(
            replay_path,
            world,
            event_bus,
            metrics,
            communication,
            run_metadata,
            snapshots,
            run_summary,
        )
        return SimulationResult(
            world=world,
            metrics=metrics,
            communication=communication,
            run_summary=run_summary,
            metadata=run_metadata,
            replay_path=replay_path,
        )

    def _should_continue(self, world: WorldState) -> bool:
        alive_agents = self._alive_agent_count(world)
        if self.config.stop_when_one_agent_remains:
            return alive_agents > 1
        return alive_agents > 0

    def _alive_agent_count(self, world: WorldState) -> int:
        return sum(1 for agent in world.agents.values() if agent.alive)

    def _build_run_metadata(self, world: WorldState, agents: dict[str, BaseAgent]) -> dict[str, object]:
        family_ids = sorted({agent.family_id for agent in world.agents.values()})
        run_id = self._build_run_id()
        return {
            "run_id": run_id,
            "experiment_id": self.config.experiment_id,
            "run_group": self.config.run_group,
            "benchmark_group": self.config.benchmark_group,
            "generation": self.config.generation,
            "run_index": self.config.run_index,
            "lineage_id": self.config.lineage_id,
            "mode": self.config.mode,
            "seed": self.config.random_seed,
            "memory_enabled": self.config.memory_enabled,
            "trust_enabled": self.config.trust_enabled,
            "social_reasoning_enabled": self.config.social_reasoning_enabled,
            "planning_enabled": self.config.planning_enabled,
            "family_ids": family_ids,
            "agent_traits": {agent_id: agent.trait.value for agent_id, agent in sorted(agents.items())},
        }

    def _build_run_id(self) -> str:
        return (
            f"{self.config.experiment_id}"
            f"-{self.config.run_group}"
            f"-g{self.config.generation:03d}"
            f"-r{self.config.run_index:03d}"
            f"-{self.config.mode}"
            f"-s{self.config.random_seed}"
        )

    def _build_replay_path(
        self,
        run_metadata: dict[str, object],
        replay_name: str,
        archive_run: bool,
    ) -> Path:
        if not archive_run:
            return self.config.artifact_dir / replay_name

        generation_dir = f"generation_{int(run_metadata['generation']):03d}"
        run_dir = f"run_{int(run_metadata['run_index']):03d}_{run_metadata['run_id']}"
        return self.config.artifact_dir / self.config.experiment_id / self.config.run_group / generation_dir / run_dir / replay_name

    def _build_snapshot(self, world: WorldState, event_bus: EventBus) -> dict[str, object]:
        return {
            "turn": world.turn,
            "event_count": len(event_bus.events),
            "world": serialize_world_state(world),
            "trust": self.trust_tracker.serialize_scores(),
        }

    def _prepare_pending_decisions(
        self,
        world: WorldState,
        agents: dict[str, BaseAgent],
        event_bus: EventBus,
        position_history: dict[str, list[Position]],
        resource_history: dict[str, dict[tuple[int, int, str], dict[str, int]]],
    ) -> list[PendingDecision]:
        pending_decisions: list[PendingDecision] = []
        for agent_id in sorted(agents):
            if not world.agents[agent_id].alive:
                continue
            social_hints = (
                self.trust_tracker.get_hints(agent_id, world.agents[agent_id].family_id, world.turn)
                if self.config.social_reasoning_enabled
                else []
            )
            recent_positions = list(position_history.get(agent_id, []))[-self.config.recent_event_limit :]
            explored_positions = self._unique_position_history(position_history.get(agent_id, []))
            nearby_unexplored_positions = self._nearby_unexplored_positions(world, explored_positions, world.agents[agent_id].position)
            exploration_ratio = len(explored_positions) / max(world.width * world.height, 1)
            observation = build_observation(
                world,
                agent_id,
                radius=self.config.observation_radius,
                social_hints=social_hints,
            )
            self._update_resource_history(resource_history, agent_id, observation.visible_resources, world.turn)
            observation = build_observation(
                world,
                agent_id,
                radius=self.config.observation_radius,
                social_hints=social_hints,
                recent_self_events=self._recent_agent_events(event_bus, agent_id),
                recent_received_messages=(
                    self._recent_received_messages(event_bus, agent_id, world.agents[agent_id].family_id)
                    if self.config.social_reasoning_enabled
                    else []
                ),
                recent_positions=recent_positions,
                explored_positions=explored_positions,
                nearby_unexplored_positions=nearby_unexplored_positions,
                resource_hotspots=self._resource_hotspots(resource_history, agent_id),
                agent_profiles=self._agent_profiles(event_bus, world, agent_id) if self.config.social_reasoning_enabled else [],
                unique_positions_visited=len({(position.x, position.y) for position in position_history.get(agent_id, [])}),
                exploration_ratio=round(exploration_ratio, 4),
            )
            if self.config.memory_enabled:
                memory_package = self.retrieval_engine.get_memory_package(
                    observation,
                    family_id=world.agents[agent_id].family_id,
                    lineage_id=self.config.lineage_id,
                )
                agents[agent_id].set_memory_context(memory_package)
            pending_decisions.append(PendingDecision(agent_id=agent_id, observation=observation))
        return pending_decisions

    def _append_position_history(self, world: WorldState, position_history: dict[str, list[Position]]) -> None:
        for agent_id, agent in world.agents.items():
            position_history.setdefault(agent_id, []).append(agent.position)

    def _unique_position_history(self, positions: list[Position]) -> list[Position]:
        unique_positions: list[Position] = []
        seen: set[tuple[int, int]] = set()
        for position in positions:
            key = (position.x, position.y)
            if key in seen:
                continue
            seen.add(key)
            unique_positions.append(position)
        return unique_positions

    def _nearby_unexplored_positions(
        self,
        world: WorldState,
        explored_positions: list[Position],
        current_position: Position,
        radius: int = 2,
    ) -> list[Position]:
        explored = {(position.x, position.y) for position in explored_positions}
        candidates: list[Position] = []
        for delta_y in range(-radius, radius + 1):
            for delta_x in range(-radius, radius + 1):
                x_coord = current_position.x + delta_x
                y_coord = current_position.y + delta_y
                if x_coord < 0 or y_coord < 0 or x_coord >= world.width or y_coord >= world.height:
                    continue
                if (x_coord, y_coord) in explored:
                    continue
                candidates.append(Position(x_coord, y_coord))
        candidates.sort(key=lambda position: abs(position.x - current_position.x) + abs(position.y - current_position.y))
        return candidates[:8]

    def _update_resource_history(
        self,
        resource_history: dict[str, dict[tuple[int, int, str], dict[str, int]]],
        agent_id: str,
        visible_resources: dict[Position, object],
        turn: int,
    ) -> None:
        agent_history = resource_history.setdefault(agent_id, {})
        for position, resource in visible_resources.items():
            key = (position.x, position.y, resource.kind.value)
            record = agent_history.setdefault(key, {"sightings": 0, "last_seen_turn": turn})
            record["sightings"] += 1
            record["last_seen_turn"] = turn

    def _resource_hotspots(
        self,
        resource_history: dict[str, dict[tuple[int, int, str], dict[str, int]]],
        agent_id: str,
    ) -> list[ResourceHotspot]:
        hotspots = [
            ResourceHotspot(
                position=Position(x_coord, y_coord),
                resource_kind=resource_kind,
                sightings=int(record["sightings"]),
                last_seen_turn=int(record["last_seen_turn"]),
            )
            for (x_coord, y_coord, resource_kind), record in resource_history.get(agent_id, {}).items()
        ]
        hotspots.sort(key=lambda hotspot: (hotspot.sightings, hotspot.last_seen_turn), reverse=True)
        return hotspots[:6]

    def _recent_agent_events(self, event_bus: EventBus, agent_id: str) -> list[RecentEventSummary]:
        recent_events = [event for event in event_bus.events if event.agent_id == agent_id][-self.config.recent_event_limit :]
        return [
            RecentEventSummary(
                turn=event.turn,
                event_type=event.event_type.value,
                position=self._position_from_payload(event.payload),
                detail=self._event_detail(event),
            )
            for event in recent_events
        ]

    def _recent_received_messages(self, event_bus: EventBus, agent_id: str, receiver_family_id: str) -> list[ReceivedMessageSummary]:
        received_messages: list[ReceivedMessageSummary] = []
        for event in event_bus.events:
            if event.event_type is not EventType.MESSAGE or event.agent_id == agent_id:
                continue
            message = event.payload.get("message", {})
            if not isinstance(message, dict):
                continue
            sender_id = str(message.get("sender_id", event.agent_id or ""))
            sender_family_id = str(message.get("sender_family_id", event.family_id or ""))
            assessment = self.trust_tracker.assess_signal(
                agent_id,
                receiver_family_id,
                sender_id,
                sender_family_id,
                str(message.get("intent", "contact")),
            )
            received_messages.append(
                ReceivedMessageSummary(
                    turn=event.turn,
                    sender_id=sender_id,
                    sender_family_id=sender_family_id,
                    content=str(message.get("content", "")),
                    intent=str(message.get("intent", "contact")),
                    resource_hint=str(message.get("resource_hint")) if message.get("resource_hint") is not None else None,
                    target_position=self._position_from_message(message),
                    trust_score=self.trust_tracker.get_score(agent_id, sender_id),
                    sender_reputation=assessment.sender_reputation,
                    message_utility=assessment.message_utility,
                    alliance_likelihood=assessment.alliance_likelihood,
                    threat_level=assessment.threat_level,
                )
            )
        return received_messages[-self.config.recent_event_limit :]

    def _agent_profiles(self, event_bus: EventBus, world: WorldState, observer_id: str) -> list[AgentSocialProfile]:
        message_counts: dict[str, int] = {}
        false_gold_counts: dict[str, int] = {}
        gold_competition_counts: dict[str, int] = {}

        for event in event_bus.events:
            if event.agent_id is None or event.agent_id == observer_id:
                continue
            if event.event_type is EventType.MESSAGE:
                message_counts[event.agent_id] = message_counts.get(event.agent_id, 0) + 1
                message = event.payload.get("message", {})
                intent = str(message.get("intent", "")) if isinstance(message, dict) else ""
                if intent == "false_gold":
                    false_gold_counts[event.agent_id] = false_gold_counts.get(event.agent_id, 0) + 1
                if intent in {"share_gold", "claim_gold", "false_gold"}:
                    gold_competition_counts[event.agent_id] = gold_competition_counts.get(event.agent_id, 0) + 1
            elif event.event_type is EventType.FORAGE and str(event.payload.get("resource")) == "gold":
                gold_competition_counts[event.agent_id] = gold_competition_counts.get(event.agent_id, 0) + 1

        profiles: list[AgentSocialProfile] = []
        for agent_id, agent in sorted(world.agents.items()):
            if agent_id == observer_id:
                continue
            truth_score = round(self.trust_tracker.get_score(observer_id, agent_id), 3)
            false_gold_count = false_gold_counts.get(agent_id, 0)
            gold_competition_count = gold_competition_counts.get(agent_id, 0)
            message_count = message_counts.get(agent_id, 0)
            sender_reputation = round(max(-2.0, min(2.0, truth_score - false_gold_count * 0.18)), 3)
            alliance_likelihood = round(
                max(-1.0, min(1.0, truth_score * 0.45 + (0.1 if message_count > 0 else 0.0) - false_gold_count * 0.35 - gold_competition_count * 0.08)),
                3,
            )
            threat_level = round(
                max(0.0, min(2.0, 0.2 + false_gold_count * 0.55 + gold_competition_count * 0.12 + max(-truth_score, 0.0) * 0.2)),
                3,
            )
            profiles.append(
                AgentSocialProfile(
                    agent_id=agent_id,
                    family_id=agent.family_id,
                    truth_score=truth_score,
                    message_count=message_count,
                    false_gold_count=false_gold_count,
                    gold_competition_count=gold_competition_count,
                    sender_reputation=sender_reputation,
                    alliance_likelihood=alliance_likelihood,
                    threat_level=threat_level,
                )
            )
        profiles.sort(
            key=lambda profile: (profile.sender_reputation, profile.alliance_likelihood, -profile.threat_level, -profile.false_gold_count),
            reverse=True,
        )
        return profiles

    def _position_from_payload(self, payload: dict[str, object]) -> Position | None:
        position = payload.get("position")
        if isinstance(position, dict) and {"x", "y"}.issubset(position):
            return Position(int(position["x"]), int(position["y"]))
        target_position = payload.get("message", {}) if isinstance(payload.get("message"), dict) else {}
        target = target_position.get("target_position")
        if isinstance(target, dict) and {"x", "y"}.issubset(target):
            return Position(int(target["x"]), int(target["y"]))
        return None

    def _position_from_message(self, message: dict[str, object]) -> Position | None:
        target = message.get("target_position")
        if isinstance(target, dict) and {"x", "y"}.issubset(target):
            return Position(int(target["x"]), int(target["y"]))
        return None

    def _event_detail(self, event) -> str:
        if event.event_type is EventType.MOVE:
            return str(event.payload.get("direction", ""))
        if event.event_type is EventType.FORAGE:
            return str(event.payload.get("resource", ""))
        if event.event_type is EventType.MESSAGE:
            message = event.payload.get("message", {})
            if isinstance(message, dict):
                return str(message.get("intent", ""))
        if event.event_type is EventType.TRUST_UPDATE:
            return str(event.payload.get("reason", ""))
        return ""

    def _collect_actions(
        self,
        agents: dict[str, BaseAgent],
        pending_decisions: list[PendingDecision],
    ) -> dict[str, AgentAction]:
        if not pending_decisions:
            return {}

        max_workers = min(max(self.config.decision_workers, 1), len(pending_decisions))
        if max_workers == 1:
            return {
                decision.agent_id: agents[decision.agent_id].choose_action(decision.observation)
                for decision in pending_decisions
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                decision.agent_id: executor.submit(agents[decision.agent_id].choose_action, decision.observation)
                for decision in pending_decisions
            }
            return {
                agent_id: future_map[agent_id].result()
                for agent_id in sorted(future_map)
            }
