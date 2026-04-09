from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MOVE = "move"
    MESSAGE = "message"
    FORAGE = "forage"
    FORAGE_MISS = "forage_miss"
    COOPERATION = "cooperation"
    TRUST_UPDATE = "trust_update"
    REST = "rest"
    DEATH = "death"


@dataclass(slots=True)
class SimulationEvent:
    turn: int
    event_type: EventType
    agent_id: str | None = None
    family_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "family_id": self.family_id,
            "payload": self.payload,
        }


@dataclass(slots=True)
class EventBus:
    events: list[SimulationEvent] = field(default_factory=list)

    def publish(self, event: SimulationEvent) -> None:
        self.events.append(event)

    def record(
        self,
        *,
        turn: int,
        event_type: EventType,
        agent_id: str | None = None,
        family_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.publish(
            SimulationEvent(
                turn=turn,
                event_type=event_type,
                agent_id=agent_id,
                family_id=family_id,
                payload=payload or {},
            )
        )

    def to_serializable(self) -> list[dict[str, Any]]:
        return [event.to_serializable() for event in self.events]
