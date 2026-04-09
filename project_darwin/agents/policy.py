from dataclasses import dataclass, field
from math import exp

from project_darwin.agents.action_space import ActionType, Direction
from project_darwin.agents.traits import TraitConfig
from project_darwin.environment.observation_builder import Observation
from project_darwin.simulation.state import Position, ResourceType


@dataclass(slots=True)
class PolicyBias:
    action_weights: dict[ActionType, float] = field(
        default_factory=lambda: {
            ActionType.REST: 0.0,
            ActionType.MOVE: 0.0,
            ActionType.MESSAGE: 0.0,
            ActionType.FORAGE: 0.0,
        }
    )

    def update_reward(self, action_type: ActionType, reward: float) -> None:
        self.action_weights[action_type] = self.action_weights.get(action_type, 0.0) + reward


@dataclass(frozen=True, slots=True)
class ObservationFeatures:
    energy: int
    energy_low: bool
    on_resource: bool
    visible_food_count: int
    visible_gold_count: int
    nearby_agent_count: int
    recent_message_count: int
    known_hotspot_count: int
    exploration_ratio: float


def extract_features(observation: Observation) -> ObservationFeatures:
    visible_food_count = sum(
        1 for resource in observation.visible_resources.values() if resource.kind is ResourceType.FOOD
    )
    visible_gold_count = sum(
        1 for resource in observation.visible_resources.values() if resource.kind is ResourceType.GOLD
    )
    return ObservationFeatures(
        energy=observation.self_state.energy,
        energy_low=observation.self_state.energy <= 3,
        on_resource=observation.self_state.position in observation.visible_resources,
        visible_food_count=visible_food_count,
        visible_gold_count=visible_gold_count,
        nearby_agent_count=len(observation.nearby_agents),
        recent_message_count=len(observation.recent_received_messages),
        known_hotspot_count=len(observation.resource_hotspots),
        exploration_ratio=observation.exploration_ratio,
    )


def get_action_scores(observation: Observation, trait_config: TraitConfig) -> dict[ActionType, float]:
    features = extract_features(observation)
    scores = dict(trait_config.base_action_bias)

    if features.energy_low:
        _apply_bias(scores, trait_config.low_energy_bias)
    if features.visible_food_count > 0:
        _apply_bias(scores, trait_config.food_visible_bias)
    if features.visible_gold_count > 0:
        _apply_bias(scores, trait_config.gold_visible_bias)
    if features.nearby_agent_count > 0:
        _apply_bias(scores, trait_config.nearby_agent_bias)
    if features.on_resource:
        scores[ActionType.FORAGE] += 1.8

    return scores


def apply_policy_bias(
    action_scores: dict[ActionType, float], policy_bias: PolicyBias
) -> dict[ActionType, float]:
    merged_scores = dict(action_scores)
    for action_type, bonus in policy_bias.action_weights.items():
        merged_scores[action_type] = merged_scores.get(action_type, 0.0) + bonus
    return merged_scores


def choose_move_direction(observation: Observation, trait_config: TraitConfig) -> Direction:
    current_position = observation.self_state.position
    target = _select_target_position(observation, trait_config)
    if target is None:
        cycle = trait_config.exploration_cycle
        return cycle[observation.turn % len(cycle)]

    delta_x = target.x - current_position.x
    delta_y = target.y - current_position.y
    if abs(delta_x) >= abs(delta_y):
        return Direction.RIGHT if delta_x > 0 else Direction.LEFT
    return Direction.DOWN if delta_y > 0 else Direction.UP


def sample_action(
    action_scores: dict[ActionType, float], temperature: float, random_value: float
) -> ActionType:
    safe_temperature = max(temperature, 0.1)
    max_score = max(action_scores.values())

    # Convert heuristic scores into a soft distribution so agents keep tendencies instead of rigid scripts.
    weights = {
        action_type: exp((score - max_score) / safe_temperature)
        for action_type, score in action_scores.items()
    }
    total_weight = sum(weights.values())
    threshold = random_value * total_weight

    cumulative_weight = 0.0
    for action_type, weight in weights.items():
        cumulative_weight += weight
        if threshold <= cumulative_weight:
            return action_type

    return ActionType.REST


def _select_target_position(observation: Observation, trait_config: TraitConfig) -> Position | None:
    prioritized_positions: list[Position] = []
    for resource_type in trait_config.resource_priority:
        resource_positions = [
            position
            for position, resource in observation.visible_resources.items()
            if resource.kind is resource_type
        ]
        if resource_positions:
            prioritized_positions.extend(sorted(resource_positions, key=lambda pos: _manhattan(observation.self_state.position, pos)))

    if prioritized_positions:
        return prioritized_positions[0]
    if observation.resource_hotspots:
        return observation.resource_hotspots[0].position
    if observation.nearby_unexplored_positions:
        return observation.nearby_unexplored_positions[0]
    if observation.nearby_agents:
        return min(
            (agent.position for agent in observation.nearby_agents),
            key=lambda pos: _manhattan(observation.self_state.position, pos),
        )
    return None


def _apply_bias(base_scores: dict[ActionType, float], additional_bias: dict[ActionType, float]) -> None:
    for action_type, bias in additional_bias.items():
        base_scores[action_type] = base_scores.get(action_type, 0.0) + bias


def _manhattan(source: Position, target: Position) -> int:
    return abs(source.x - target.x) + abs(source.y - target.y)
