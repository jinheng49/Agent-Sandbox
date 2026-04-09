from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from project_darwin.agents.action_space import ActionType, AgentAction, Direction
from project_darwin.agents.policy import PolicyBias
from project_darwin.agents.traits import TraitProfile
from project_darwin.environment.observation_builder import Observation


@dataclass(slots=True)
class BaseAgent(ABC):
    agent_id: str
    family_id: str
    trait: TraitProfile
    policy_bias: PolicyBias = field(default_factory=PolicyBias)

    @abstractmethod
    def choose_action(self, observation: Observation) -> AgentAction:
        raise NotImplementedError

    def set_memory_context(self, memories: Any) -> None:
        """Agents that use lineage memories can override this hook."""
        return None


@dataclass(slots=True)
class ScriptedSurvivor(BaseAgent):
    def choose_action(self, observation: Observation) -> AgentAction:
        own_position = observation.self_state.position

        if own_position in observation.visible_resources:
            return AgentAction(action_type=ActionType.FORAGE)

        if observation.self_state.energy <= 3 and self.trait is TraitProfile.SILENT:
            return AgentAction(action_type=ActionType.REST)

        if observation.nearby_agents and self.trait is not TraitProfile.SILENT:
            message = "g" if self.trait is TraitProfile.GREEDY else "c"
            return AgentAction(action_type=ActionType.MESSAGE, content=message)

        next_direction = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP][observation.turn % 4]
        return AgentAction(action_type=ActionType.MOVE, direction=next_direction)
