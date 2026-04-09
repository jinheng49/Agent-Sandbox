import random
from dataclasses import dataclass, field
from typing import Any

from project_darwin.agents.action_space import ActionType, AgentAction, Direction, MessageIntent, ShortTermPlan
from project_darwin.agents.base_agent import BaseAgent
from project_darwin.agents.policy import apply_policy_bias, choose_move_direction, get_action_scores, sample_action
from project_darwin.agents.traits import get_trait_config
from project_darwin.environment.observation_builder import Observation, SocialHint
from project_darwin.memory.retrieval_engine import MemoryContextPackage, MemoryDirective, coerce_memory_package
from project_darwin.simulation.state import Position, ResourceType


@dataclass(slots=True)
class HeuristicSurvivor(BaseAgent):
    seed: int = 0
    temperature: float = 1.0
    social_reasoning_enabled: bool = True
    planning_enabled: bool = True
    memory_context: list[str] = field(default_factory=list)
    memory_package: MemoryContextPackage = field(default_factory=MemoryContextPackage)
    plan_context: ShortTermPlan = field(default_factory=ShortTermPlan)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose_action(self, observation: Observation) -> AgentAction:
        trait_config = get_trait_config(self.trait)
        action_scores = get_action_scores(observation, trait_config)
        self._apply_memory_bias(action_scores)
        self._apply_social_bias(action_scores, observation)
        action_scores = apply_policy_bias(action_scores, self.policy_bias)
        action_type = sample_action(action_scores, self.temperature, self._rng.random())

        if action_type is ActionType.MOVE:
            direction = (
                self._choose_plan_direction(observation)
                or
                self._choose_memory_guided_direction(observation)
                or
                self._choose_social_direction(observation)
                or self._choose_hotspot_direction(observation)
                or self._choose_unexplored_direction(observation)
                or self._choose_loop_break_direction(observation)
                or choose_move_direction(observation, trait_config)
            )
            return AgentAction(action_type=ActionType.MOVE, direction=direction)

        if action_type is ActionType.MESSAGE:
            return self._build_message_action(observation, trait_config.message_token)

        if action_type is ActionType.FORAGE:
            resource = observation.visible_resources.get(observation.self_state.position)
            share_gold = (
                resource is not None
                and resource.kind is ResourceType.GOLD
                and self.trait.value == "cooperative"
                and bool(observation.nearby_agents)
            )
            return AgentAction(action_type=ActionType.FORAGE, share_with_nearby=share_gold)

        return AgentAction(action_type=action_type)

    def set_memory_context(self, memories: Any) -> None:
        package = coerce_memory_package(memories)
        self.memory_package = package
        self.memory_context = list(package.typed_lessons or package.soft_hints)

    def set_plan_context(self, plan: ShortTermPlan | None) -> None:
        self.plan_context = plan or ShortTermPlan()

    def _build_message_action(self, observation: Observation, token: str) -> AgentAction:
        gold_positions = [
            position
            for position, resource in observation.visible_resources.items()
            if resource.kind is ResourceType.GOLD
        ]
        food_positions = [
            position
            for position, resource in observation.visible_resources.items()
            if resource.kind is ResourceType.FOOD
        ]

        if gold_positions and observation.nearby_agents:
            nearest_gold = min(
                gold_positions,
                key=lambda position: abs(position.x - observation.self_state.position.x) + abs(position.y - observation.self_state.position.y),
            )
            if self.trait.value == "cooperative":
                return AgentAction(
                    action_type=ActionType.MESSAGE,
                    content=f"{token}g",
                    message_intent=MessageIntent.SHARE_GOLD,
                    message_target=(nearest_gold.x, nearest_gold.y),
                    resource_hint=ResourceType.GOLD.value,
                )
            return AgentAction(
                action_type=ActionType.MESSAGE,
                content=f"{token}!",
                message_intent=MessageIntent.CLAIM_GOLD,
                message_target=(nearest_gold.x, nearest_gold.y),
                resource_hint=ResourceType.GOLD.value,
            )

        if food_positions and observation.nearby_agents and self.trait.value == "cooperative":
            nearest_food = min(
                food_positions,
                key=lambda position: abs(position.x - observation.self_state.position.x) + abs(position.y - observation.self_state.position.y),
            )
            return AgentAction(
                action_type=ActionType.MESSAGE,
                content=f"{token}f",
                message_intent=MessageIntent.SHARE_FOOD,
                message_target=(nearest_food.x, nearest_food.y),
                resource_hint=ResourceType.FOOD.value,
            )

        if self.trait.value == "greedy" and observation.nearby_agents and gold_positions == []:
            fake_target = food_positions[0] if food_positions else observation.self_state.position
            return AgentAction(
                action_type=ActionType.MESSAGE,
                content=f"{token}x",
                message_intent=MessageIntent.FALSE_GOLD,
                message_target=(fake_target.x, fake_target.y),
                resource_hint=ResourceType.GOLD.value,
            )

        return AgentAction(action_type=ActionType.MESSAGE, content=token, message_intent=MessageIntent.CONTACT)

    def _apply_social_bias(self, action_scores: dict[ActionType, float], observation: Observation) -> None:
        if not self.social_reasoning_enabled:
            return
        best_hint = self._best_hint(observation)
        if best_hint is None:
            if observation.resource_hotspots:
                action_scores[ActionType.MOVE] += 0.35
            elif observation.nearby_unexplored_positions:
                action_scores[ActionType.MOVE] += 0.25
            return
        if best_hint.threat_level >= 0.8:
            action_scores[ActionType.MESSAGE] -= 0.5
            action_scores[ActionType.MOVE] -= 0.15
            action_scores[ActionType.REST] += 0.1
        if best_hint.alliance_likelihood >= 0.4:
            action_scores[ActionType.MESSAGE] += 0.45
            action_scores[ActionType.FORAGE] += 0.35
        if best_hint.message_utility >= 0.35:
            action_scores[ActionType.MOVE] += 0.45
        if best_hint.target_position == observation.self_state.position:
            action_scores[ActionType.FORAGE] += 1.1
            return
        action_scores[ActionType.MOVE] += 0.9

    def _choose_social_direction(self, observation: Observation):
        if not self.social_reasoning_enabled:
            return None
        if self._blocks_suspicious_reroute(observation):
            return None
        best_hint = self._best_hint(observation)
        if best_hint is None or best_hint.target_position is None:
            return None
        current_position = observation.self_state.position
        target = best_hint.target_position
        if target == current_position:
            return None
        delta_x = target.x - current_position.x
        delta_y = target.y - current_position.y
        if abs(delta_x) >= abs(delta_y):
            return self._direction_from_delta(delta_x, horizontal=True)
        return self._direction_from_delta(delta_y, horizontal=False)

    def _best_hint(self, observation: Observation) -> SocialHint | None:
        if not self.social_reasoning_enabled:
            return None
        trusted_hints = [hint for hint in observation.social_hints if hint.trust_score >= 0.0 and hint.target_position is not None]
        if not trusted_hints:
            return None
        profiles = {profile.agent_id: profile for profile in observation.agent_profiles}
        return max(
            trusted_hints,
            key=lambda hint: (
                hint.message_utility,
                hint.sender_reputation,
                hint.alliance_likelihood,
                -hint.threat_level,
                hint.trust_score,
                profiles.get(hint.sender_id).truth_score if hint.sender_id in profiles else 0.0,
                hint.intent == MessageIntent.SHARE_GOLD.value,
                -(profiles.get(hint.sender_id).false_gold_count if hint.sender_id in profiles else 0),
            ),
        )

    def _choose_hotspot_direction(self, observation: Observation) -> Direction | None:
        if observation.visible_resources:
            return None
        current_position = observation.self_state.position
        for hotspot in observation.resource_hotspots:
            if hotspot.position == current_position:
                continue
            return self._step_toward(current_position, hotspot.position)
        return None

    def _choose_unexplored_direction(self, observation: Observation) -> Direction | None:
        if observation.visible_resources or not observation.nearby_unexplored_positions:
            return None
        return self._step_toward(observation.self_state.position, observation.nearby_unexplored_positions[0])

    def _choose_loop_break_direction(self, observation: Observation) -> Direction | None:
        recent_positions = observation.recent_positions
        if len(recent_positions) < 4:
            return None

        last_four = recent_positions[-4:]
        if last_four[0] != last_four[2] or last_four[1] != last_four[3] or last_four[0] == last_four[1]:
            return None

        horizontal_loop = last_four[0].y == last_four[1].y and last_four[0].x != last_four[1].x
        vertical_loop = last_four[0].x == last_four[1].x and last_four[0].y != last_four[1].y
        if horizontal_loop:
            return Direction.DOWN if observation.turn % 2 == 0 else Direction.UP
        if vertical_loop:
            return Direction.RIGHT if observation.turn % 2 == 0 else Direction.LEFT
        return None

    def _direction_from_delta(self, delta: int, *, horizontal: bool):
        if horizontal:
            return choose_horizontal_direction(delta)
        return choose_vertical_direction(delta)

    def _step_toward(self, current_position, target_position) -> Direction | None:
        if target_position == current_position:
            return None
        delta_x = target_position.x - current_position.x
        delta_y = target_position.y - current_position.y
        if abs(delta_x) >= abs(delta_y):
            return self._direction_from_delta(delta_x, horizontal=True)
        return self._direction_from_delta(delta_y, horizontal=False)

    def _apply_memory_bias(self, action_scores: dict[ActionType, float]) -> None:
        for directive in self.memory_package.directives:
            for action_name, bonus in directive.action_bias.items():
                action_scores[ActionType(action_name)] += bonus

        for memory in self.memory_context:
            lowered = memory.lower()
            if "reduce broadcasts" in lowered or "overcommunicated" in lowered:
                action_scores[ActionType.MESSAGE] -= 1.4
                action_scores[ActionType.REST] += 0.3
                action_scores[ActionType.FORAGE] += 0.4
            if "seek nearby resources" in lowered or "missed_resources" in lowered:
                action_scores[ActionType.FORAGE] += 1.0
                action_scores[ActionType.MOVE] += 0.2
            if "cooperate" in lowered or "share gold" in lowered:
                action_scores[ActionType.MESSAGE] += 0.7
                action_scores[ActionType.FORAGE] += 0.5

    def _choose_memory_guided_direction(self, observation: Observation) -> Direction | None:
        current_position = observation.self_state.position
        for directive in self.memory_package.directives:
            if directive.target_preference == "cooperate_on_gold" and observation.nearby_agents:
                gold_positions = [
                    position
                    for position, resource in observation.visible_resources.items()
                    if resource.kind is ResourceType.GOLD
                ]
                if gold_positions:
                    return self._step_toward(current_position, gold_positions[0])
            if directive.target_preference in {"seek_resource", "seek_verified_resource"}:
                if observation.resource_hotspots:
                    return self._step_toward(current_position, observation.resource_hotspots[0].position)
                if observation.visible_resources:
                    target = min(
                        observation.visible_resources,
                        key=lambda position: abs(position.x - current_position.x) + abs(position.y - current_position.y),
                    )
                    return self._step_toward(current_position, target)
        return None

    def _choose_plan_direction(self, observation: Observation) -> Direction | None:
        if not self.planning_enabled:
            return None
        if self.plan_context.is_empty() or self.plan_context.planned_target_position is None:
            return None
        target_x, target_y = self.plan_context.planned_target_position
        return self._step_toward(observation.self_state.position, Position(target_x, target_y))

    def _blocks_suspicious_reroute(self, observation: Observation) -> bool:
        if not self.social_reasoning_enabled:
            return False
        if not observation.recent_received_messages:
            return False
        return any(
            directive.caution_against in {"suspicious_signal", "false_gold"}
            for directive in self.memory_package.directives
        ) and any(
            message.trust_score < 0 or message.threat_level >= 0.75 or message.message_utility < -0.2
            for message in observation.recent_received_messages
        )


def choose_horizontal_direction(delta_x: int):
    from project_darwin.agents.action_space import Direction

    return Direction.RIGHT if delta_x > 0 else Direction.LEFT


def choose_vertical_direction(delta_y: int):
    from project_darwin.agents.action_space import Direction

    return Direction.DOWN if delta_y > 0 else Direction.UP