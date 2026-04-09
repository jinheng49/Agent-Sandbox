from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceType(str, Enum):
    FOOD = "food"
    GOLD = "gold"


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int


@dataclass(slots=True)
class ResourceNode:
    kind: ResourceType
    amount: int = 1


@dataclass(slots=True)
class AgentState:
    agent_id: str
    family_id: str
    position: Position
    energy: int
    alive: bool = True
    inventory: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class WorldState:
    turn: int
    width: int
    height: int
    agents: dict[str, AgentState]
    resources: dict[Position, ResourceNode]


def serialize_world_state(world: WorldState) -> dict[str, Any]:
    return {
        "turn": world.turn,
        "width": world.width,
        "height": world.height,
        "agents": {
            agent_id: {
                "agent_id": agent.agent_id,
                "family_id": agent.family_id,
                "energy": agent.energy,
                "alive": agent.alive,
                "position": {"x": agent.position.x, "y": agent.position.y},
                "inventory": dict(agent.inventory),
            }
            for agent_id, agent in sorted(world.agents.items())
        },
        "resources": [
            {
                "position": {"x": position.x, "y": position.y},
                "kind": resource.kind.value,
                "amount": resource.amount,
            }
            for position, resource in sorted(world.resources.items(), key=lambda item: (item[0].y, item[0].x))
        ],
    }


def deserialize_world_state(snapshot: dict[str, Any]) -> WorldState:
    agents = {
        agent_id: AgentState(
            agent_id=agent_state["agent_id"],
            family_id=agent_state["family_id"],
            position=Position(agent_state["position"]["x"], agent_state["position"]["y"]),
            energy=int(agent_state["energy"]),
            alive=bool(agent_state["alive"]),
            inventory=dict(agent_state.get("inventory", {})),
        )
        for agent_id, agent_state in snapshot.get("agents", {}).items()
    }
    resources = {
        Position(resource["position"]["x"], resource["position"]["y"]): ResourceNode(
            kind=ResourceType(resource["kind"]),
            amount=int(resource.get("amount", 1)),
        )
        for resource in snapshot.get("resources", [])
    }
    return WorldState(
        turn=int(snapshot.get("turn", 0)),
        width=int(snapshot.get("width", 0)),
        height=int(snapshot.get("height", 0)),
        agents=agents,
        resources=resources,
    )
