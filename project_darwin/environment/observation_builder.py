from dataclasses import dataclass, field

from project_darwin.simulation.state import AgentState, Position, ResourceNode, WorldState


@dataclass(frozen=True, slots=True)
class SocialHint:
    sender_id: str
    sender_family_id: str
    intent: str
    resource_hint: str | None
    target_position: Position | None
    trust_score: float
    sender_reputation: float = 0.0
    message_utility: float = 0.0
    alliance_likelihood: float = 0.0
    threat_level: float = 0.0


@dataclass(frozen=True, slots=True)
class RecentEventSummary:
    turn: int
    event_type: str
    position: Position | None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ReceivedMessageSummary:
    turn: int
    sender_id: str
    sender_family_id: str
    content: str
    intent: str
    resource_hint: str | None
    target_position: Position | None
    trust_score: float
    sender_reputation: float = 0.0
    message_utility: float = 0.0
    alliance_likelihood: float = 0.0
    threat_level: float = 0.0


@dataclass(frozen=True, slots=True)
class ResourceHotspot:
    position: Position
    resource_kind: str
    sightings: int
    last_seen_turn: int


@dataclass(frozen=True, slots=True)
class AgentSocialProfile:
    agent_id: str
    family_id: str
    truth_score: float
    message_count: int
    false_gold_count: int
    gold_competition_count: int
    sender_reputation: float = 0.0
    alliance_likelihood: float = 0.0
    threat_level: float = 0.0


@dataclass(frozen=True, slots=True)
class Observation:
    turn: int
    self_state: AgentState
    visible_resources: dict[Position, ResourceNode]
    nearby_agents: list[AgentState]
    social_hints: list[SocialHint] = field(default_factory=list)
    recent_self_events: list[RecentEventSummary] = field(default_factory=list)
    recent_received_messages: list[ReceivedMessageSummary] = field(default_factory=list)
    recent_positions: list[Position] = field(default_factory=list)
    explored_positions: list[Position] = field(default_factory=list)
    nearby_unexplored_positions: list[Position] = field(default_factory=list)
    resource_hotspots: list[ResourceHotspot] = field(default_factory=list)
    agent_profiles: list[AgentSocialProfile] = field(default_factory=list)
    unique_positions_visited: int = 0
    exploration_ratio: float = 0.0


def build_observation(
    world: WorldState,
    agent_id: str,
    radius: int = 1,
    social_hints: list[SocialHint] | None = None,
    recent_self_events: list[RecentEventSummary] | None = None,
    recent_received_messages: list[ReceivedMessageSummary] | None = None,
    recent_positions: list[Position] | None = None,
    explored_positions: list[Position] | None = None,
    nearby_unexplored_positions: list[Position] | None = None,
    resource_hotspots: list[ResourceHotspot] | None = None,
    agent_profiles: list[AgentSocialProfile] | None = None,
    unique_positions_visited: int = 0,
    exploration_ratio: float = 0.0,
) -> Observation:
    self_state = world.agents[agent_id]
    nearby_agents: list[AgentState] = []
    visible_resources: dict[Position, ResourceNode] = {}

    for other_id, other_state in world.agents.items():
        if other_id == agent_id or not other_state.alive:
            continue
        if abs(other_state.position.x - self_state.position.x) <= radius and abs(other_state.position.y - self_state.position.y) <= radius:
            nearby_agents.append(other_state)

    for position, resource in world.resources.items():
        if abs(position.x - self_state.position.x) <= radius and abs(position.y - self_state.position.y) <= radius:
            visible_resources[position] = resource

    return Observation(
        turn=world.turn,
        self_state=self_state,
        visible_resources=visible_resources,
        nearby_agents=nearby_agents,
        social_hints=list(social_hints or []),
        recent_self_events=list(recent_self_events or []),
        recent_received_messages=list(recent_received_messages or []),
        recent_positions=list(recent_positions or []),
        explored_positions=list(explored_positions or []),
        nearby_unexplored_positions=list(nearby_unexplored_positions or []),
        resource_hotspots=list(resource_hotspots or []),
        agent_profiles=list(agent_profiles or []),
        unique_positions_visited=unique_positions_visited,
        exploration_ratio=exploration_ratio,
    )
