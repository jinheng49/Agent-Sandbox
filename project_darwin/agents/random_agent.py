import random
from dataclasses import dataclass, field

from project_darwin.agents.action_space import ActionType, AgentAction, Direction
from project_darwin.agents.base_agent import BaseAgent
from project_darwin.agents.traits import TraitProfile
from project_darwin.environment.observation_builder import Observation


@dataclass(slots=True)
class RandomSurvivor(BaseAgent):
    seed: int = 0
    forage_probability: float = 0.85
    message_probability: float = 0.35
    rest_probability_low_energy: float = 0.65
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose_action(self, observation: Observation) -> AgentAction:
        own_position = observation.self_state.position

        if own_position in observation.visible_resources and self._rng.random() <= self.forage_probability:
            return AgentAction(action_type=ActionType.FORAGE)

        if observation.self_state.energy <= 3 and self._rng.random() <= self.rest_probability_low_energy:
            return AgentAction(action_type=ActionType.REST)

        if observation.nearby_agents and self._rng.random() <= self.message_probability:
            return AgentAction(action_type=ActionType.MESSAGE, content=self._message_token())

        return AgentAction(action_type=ActionType.MOVE, direction=self._rng.choice(list(Direction)))

    def _message_token(self) -> str:
        if self.trait is TraitProfile.GREEDY:
            return "g"
        if self.trait is TraitProfile.COOPERATIVE:
            return "c"
        return "s"
