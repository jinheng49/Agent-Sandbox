import json
from pathlib import Path
from typing import Any

from project_darwin.analytics.communication_analysis import CommunicationReport
from project_darwin.analytics.metrics_engine import MetricsSnapshot, RunSummary
from project_darwin.simulation.event_bus import EventBus
from project_darwin.simulation.state import WorldState, serialize_world_state


class ReplayStore:
    def write_run(
        self,
        output_path: Path,
        world: WorldState,
        event_bus: EventBus,
        metrics: MetricsSnapshot,
        communication: CommunicationReport,
        metadata: dict[str, Any],
        snapshots: list[dict[str, Any]],
        run_summary: RunSummary,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        world_snapshot = serialize_world_state(world)
        payload: dict[str, Any] = {
            "metadata": metadata,
            "turn": world.turn,
            "world": world_snapshot,
            "agents": world_snapshot["agents"],
            "resources": world_snapshot["resources"],
            "resources_remaining": len(world.resources),
            "events": event_bus.to_serializable(),
            "snapshots": snapshots,
            "metrics": {
                "turn": metrics.turn,
                "alive_agents": metrics.alive_agents,
                "total_messages": metrics.total_messages,
                "total_message_cost": metrics.total_message_cost,
                "total_forage_events": metrics.total_forage_events,
                "total_cooperation_events": metrics.total_cooperation_events,
                "total_false_gold_messages": metrics.total_false_gold_messages,
                "total_trust_updates": metrics.total_trust_updates,
                "resources_remaining": metrics.resources_remaining,
            },
            "communication": {
                "vocabulary_size": communication.vocabulary_size,
                "mean_message_length": communication.mean_message_length,
                "entropy": communication.entropy,
                "word_frequency": communication.word_frequency,
                "protocol_compression_rate": communication.protocol_compression_rate,
                "deception_frequency": communication.deception_frequency,
                "share_gold_signals": communication.share_gold_signals,
                "false_gold_signals": communication.false_gold_signals,
            },
            "run_summary": run_summary.to_dict(),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
