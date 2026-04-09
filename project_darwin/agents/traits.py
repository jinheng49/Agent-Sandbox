from dataclasses import dataclass
from enum import Enum

from project_darwin.agents.action_space import ActionType, Direction
from project_darwin.simulation.state import ResourceType


class TraitProfile(str, Enum):
    GREEDY = "greedy"
    COOPERATIVE = "cooperative"
    SILENT = "silent"


@dataclass(frozen=True, slots=True)
class TraitConfig:
    base_action_bias: dict[ActionType, float]
    low_energy_bias: dict[ActionType, float]
    food_visible_bias: dict[ActionType, float]
    gold_visible_bias: dict[ActionType, float]
    nearby_agent_bias: dict[ActionType, float]
    resource_priority: tuple[ResourceType, ...]
    exploration_cycle: tuple[Direction, ...]
    message_token: str


TRAIT_LIBRARY: dict[TraitProfile, TraitConfig] = {
    TraitProfile.GREEDY: TraitConfig(
        base_action_bias={
            ActionType.REST: -0.5,
            ActionType.MOVE: 1.1,
            ActionType.MESSAGE: -0.5,
            ActionType.FORAGE: 0.4,
        },
        low_energy_bias={
            ActionType.REST: 0.4,
            ActionType.MOVE: -0.6,
            ActionType.MESSAGE: -0.8,
            ActionType.FORAGE: 1.0,
        },
        food_visible_bias={
            ActionType.REST: -0.8,
            ActionType.MOVE: 0.1,
            ActionType.MESSAGE: -0.5,
            ActionType.FORAGE: 1.7,
        },
        gold_visible_bias={
            ActionType.REST: -1.0,
            ActionType.MOVE: 1.2,
            ActionType.MESSAGE: -0.9,
            ActionType.FORAGE: 2.4,
        },
        nearby_agent_bias={
            ActionType.REST: -0.2,
            ActionType.MOVE: 0.5,
            ActionType.MESSAGE: -0.8,
            ActionType.FORAGE: 0.3,
        },
        resource_priority=(ResourceType.GOLD, ResourceType.FOOD),
        exploration_cycle=(Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP),
        message_token="g",
    ),
    TraitProfile.COOPERATIVE: TraitConfig(
        base_action_bias={
            ActionType.REST: -0.2,
            ActionType.MOVE: 0.8,
            ActionType.MESSAGE: 0.5,
            ActionType.FORAGE: 0.5,
        },
        low_energy_bias={
            ActionType.REST: 0.6,
            ActionType.MOVE: -0.5,
            ActionType.MESSAGE: -0.3,
            ActionType.FORAGE: 0.9,
        },
        food_visible_bias={
            ActionType.REST: -0.7,
            ActionType.MOVE: 0.0,
            ActionType.MESSAGE: 0.3,
            ActionType.FORAGE: 1.6,
        },
        gold_visible_bias={
            ActionType.REST: -0.6,
            ActionType.MOVE: 0.8,
            ActionType.MESSAGE: 1.2,
            ActionType.FORAGE: 1.4,
        },
        nearby_agent_bias={
            ActionType.REST: -0.2,
            ActionType.MOVE: 0.2,
            ActionType.MESSAGE: 1.3,
            ActionType.FORAGE: 0.2,
        },
        resource_priority=(ResourceType.FOOD, ResourceType.GOLD),
        exploration_cycle=(Direction.DOWN, Direction.RIGHT, Direction.UP, Direction.LEFT),
        message_token="c",
    ),
    TraitProfile.SILENT: TraitConfig(
        base_action_bias={
            ActionType.REST: 0.2,
            ActionType.MOVE: 0.7,
            ActionType.MESSAGE: -2.0,
            ActionType.FORAGE: 0.4,
        },
        low_energy_bias={
            ActionType.REST: 1.5,
            ActionType.MOVE: -0.8,
            ActionType.MESSAGE: -1.4,
            ActionType.FORAGE: 0.8,
        },
        food_visible_bias={
            ActionType.REST: -0.6,
            ActionType.MOVE: 0.0,
            ActionType.MESSAGE: -1.0,
            ActionType.FORAGE: 1.8,
        },
        gold_visible_bias={
            ActionType.REST: -0.5,
            ActionType.MOVE: 0.9,
            ActionType.MESSAGE: -1.5,
            ActionType.FORAGE: 1.5,
        },
        nearby_agent_bias={
            ActionType.REST: 0.3,
            ActionType.MOVE: 0.5,
            ActionType.MESSAGE: -2.2,
            ActionType.FORAGE: 0.1,
        },
        resource_priority=(ResourceType.FOOD, ResourceType.GOLD),
        exploration_cycle=(Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN),
        message_token="s",
    ),
}


def get_trait_config(trait: TraitProfile) -> TraitConfig:
    return TRAIT_LIBRARY[trait]
