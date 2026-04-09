from collections import defaultdict
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

from project_darwin.analytics.communication_analysis import CommunicationReport
from project_darwin.simulation.event_bus import EventBus, EventType
from project_darwin.simulation.state import WorldState


@dataclass(slots=True)
class MetricsSnapshot:
    turn: int
    alive_agents: int
    total_messages: int
    total_message_cost: int
    total_forage_events: int
    total_cooperation_events: int
    total_false_gold_messages: int
    total_trust_updates: int
    resources_remaining: int


@dataclass(slots=True)
class RunSummary:
    run_id: str
    benchmark_group: str
    generation: int
    run_index: int
    mode: str
    memory_enabled: bool
    trust_enabled: bool
    social_reasoning_enabled: bool
    planning_enabled: bool
    lineage_id: str
    turn: int
    alive_agents: int
    average_survival_turn: float
    resource_acquisition_rate: float
    message_cost_per_turn: float
    cooperation_rate: float
    deception_frequency: float
    total_messages: int
    total_message_cost: int
    total_forage_events: int
    total_cooperation_events: int
    total_false_gold_messages: int
    total_trust_updates: int
    vocabulary_size: int
    mean_message_length: float
    entropy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark_group": self.benchmark_group,
            "generation": self.generation,
            "run_index": self.run_index,
            "mode": self.mode,
            "memory_enabled": self.memory_enabled,
            "trust_enabled": self.trust_enabled,
            "social_reasoning_enabled": self.social_reasoning_enabled,
            "planning_enabled": self.planning_enabled,
            "lineage_id": self.lineage_id,
            "turn": self.turn,
            "alive_agents": self.alive_agents,
            "average_survival_turn": self.average_survival_turn,
            "resource_acquisition_rate": self.resource_acquisition_rate,
            "message_cost_per_turn": self.message_cost_per_turn,
            "cooperation_rate": self.cooperation_rate,
            "deception_frequency": self.deception_frequency,
            "total_messages": self.total_messages,
            "total_message_cost": self.total_message_cost,
            "total_forage_events": self.total_forage_events,
            "total_cooperation_events": self.total_cooperation_events,
            "total_false_gold_messages": self.total_false_gold_messages,
            "total_trust_updates": self.total_trust_updates,
            "vocabulary_size": self.vocabulary_size,
            "mean_message_length": self.mean_message_length,
            "entropy": self.entropy,
        }


class MetricsEngine:
    def snapshot(self, world: WorldState, event_bus: EventBus) -> MetricsSnapshot:
        total_messages = 0
        total_message_cost = 0
        total_forage_events = 0
        total_cooperation_events = 0
        total_false_gold_messages = 0
        total_trust_updates = 0

        for event in event_bus.events:
            if event.event_type is EventType.MESSAGE:
                total_messages += 1
                total_message_cost += int(event.payload["cost"])
                if event.payload["message"].get("intent") == "false_gold":
                    total_false_gold_messages += 1
            elif event.event_type is EventType.FORAGE:
                total_forage_events += 1
            elif event.event_type is EventType.COOPERATION:
                total_cooperation_events += 1
            elif event.event_type is EventType.TRUST_UPDATE:
                total_trust_updates += 1

        alive_agents = sum(1 for agent in world.agents.values() if agent.alive)
        return MetricsSnapshot(
            turn=world.turn,
            alive_agents=alive_agents,
            total_messages=total_messages,
            total_message_cost=total_message_cost,
            total_forage_events=total_forage_events,
            total_cooperation_events=total_cooperation_events,
            total_false_gold_messages=total_false_gold_messages,
            total_trust_updates=total_trust_updates,
            resources_remaining=len(world.resources),
        )

    def summarize_run(
        self,
        world: WorldState,
        event_bus: EventBus,
        metrics: MetricsSnapshot,
        communication: CommunicationReport,
        metadata: dict[str, Any],
    ) -> RunSummary:
        death_turns: dict[str, int] = {}
        for event in event_bus.events:
            if event.event_type is EventType.DEATH and event.agent_id is not None:
                death_turns.setdefault(event.agent_id, event.turn)

        survival_turns = [death_turns.get(agent_id, world.turn) for agent_id in sorted(world.agents)]
        turn_count = max(metrics.turn, 1)
        average_survival_turn = sum(survival_turns) / max(len(survival_turns), 1)

        return RunSummary(
            run_id=str(metadata.get("run_id", "")),
            benchmark_group=str(metadata.get("benchmark_group", "")),
            generation=int(metadata.get("generation", 0)),
            run_index=int(metadata.get("run_index", 0)),
            mode=str(metadata.get("mode", "unknown")),
            memory_enabled=bool(metadata.get("memory_enabled", False)),
            trust_enabled=bool(metadata.get("trust_enabled", False)),
            social_reasoning_enabled=bool(metadata.get("social_reasoning_enabled", False)),
            planning_enabled=bool(metadata.get("planning_enabled", False)),
            lineage_id=str(metadata.get("lineage_id", "unknown")),
            turn=metrics.turn,
            alive_agents=metrics.alive_agents,
            average_survival_turn=round(average_survival_turn, 4),
            resource_acquisition_rate=round(metrics.total_forage_events / turn_count, 4),
            message_cost_per_turn=round(metrics.total_message_cost / turn_count, 4),
            cooperation_rate=round(metrics.total_cooperation_events / turn_count, 4),
            deception_frequency=round(metrics.total_false_gold_messages / max(metrics.total_messages, 1), 4),
            total_messages=metrics.total_messages,
            total_message_cost=metrics.total_message_cost,
            total_forage_events=metrics.total_forage_events,
            total_cooperation_events=metrics.total_cooperation_events,
            total_false_gold_messages=metrics.total_false_gold_messages,
            total_trust_updates=metrics.total_trust_updates,
            vocabulary_size=communication.vocabulary_size,
            mean_message_length=round(communication.mean_message_length, 4),
            entropy=round(communication.entropy, 4),
        )

    def summarize_generations(self, run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for summary in run_summaries:
            grouped[int(summary.get("generation", 0))].append(summary)

        generation_rows: list[dict[str, Any]] = []
        fields = [
            "turn",
            "average_survival_turn",
            "resource_acquisition_rate",
            "message_cost_per_turn",
            "cooperation_rate",
            "deception_frequency",
            "mean_message_length",
            "entropy",
            "total_messages",
            "total_message_cost",
            "total_forage_events",
            "total_cooperation_events",
            "total_false_gold_messages",
            "total_trust_updates",
            "alive_agents",
            "vocabulary_size",
        ]
        for generation in sorted(grouped):
            rows = grouped[generation]
            run_count = len(rows)
            aggregate = {
                "generation": generation,
                "run_count": run_count,
            }
            for field in fields:
                aggregate[field] = round(
                    sum(float(row.get(field, 0.0)) for row in rows) / max(run_count, 1),
                    4,
                )
            generation_rows.append(aggregate)
        return generation_rows

    def summarize_benchmark_groups(self, run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for summary in run_summaries:
            grouped[str(summary.get("benchmark_group", "unlabeled"))].append(summary)

        metric_fields = [
            "average_survival_turn",
            "resource_acquisition_rate",
            "message_cost_per_turn",
            "cooperation_rate",
            "deception_frequency",
        ]
        group_rows: list[dict[str, Any]] = []
        for benchmark_group, rows in sorted(grouped.items()):
            run_count = len(rows)
            aggregate: dict[str, Any] = {
                "benchmark_group": benchmark_group,
                "run_count": run_count,
                "mode": str(rows[0].get("mode", "unknown")),
                "memory_enabled": bool(rows[0].get("memory_enabled", False)),
                "social_reasoning_enabled": bool(rows[0].get("social_reasoning_enabled", False)),
                "planning_enabled": bool(rows[0].get("planning_enabled", False)),
            }
            for field in metric_fields:
                values = [float(row.get(field, 0.0)) for row in rows]
                aggregate[field] = round(sum(values) / max(run_count, 1), 4)
                aggregate[f"{field}_std"] = round(pstdev(values) if run_count > 1 else 0.0, 4)
            group_rows.append(aggregate)
        return group_rows
