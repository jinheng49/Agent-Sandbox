from dataclasses import dataclass, field
from typing import Any

from project_darwin.agents.action_space import ActionType, AgentAction, ShortTermPlan
from project_darwin.agents.base_agent import BaseAgent
from project_darwin.agents.cognition_graph import CognitionGraph
from project_darwin.agents.heuristic_agent import HeuristicSurvivor
from project_darwin.agents.llm_adapter import LLMAdapter
from project_darwin.environment.observation_builder import Observation
from project_darwin.memory.retrieval_engine import MemoryContextPackage, coerce_memory_package


@dataclass(slots=True)
class LLMSurvivor(BaseAgent):
    seed: int = 0
    temperature: float = 0.2
    planning_enabled: bool = True
    social_reasoning_enabled: bool = True
    llm_adapter: LLMAdapter = field(default_factory=LLMAdapter)
    memory_context: MemoryContextPackage = field(default_factory=MemoryContextPackage)
    current_plan: ShortTermPlan = field(default_factory=ShortTermPlan)
    _fallback_agent: HeuristicSurvivor = field(init=False, repr=False)
    _cognition_graph: CognitionGraph = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback_agent = HeuristicSurvivor(
            agent_id=self.agent_id,
            family_id=self.family_id,
            trait=self.trait,
            policy_bias=self.policy_bias,
            seed=self.seed,
            temperature=self.temperature,
            social_reasoning_enabled=self.social_reasoning_enabled,
            planning_enabled=self.planning_enabled,
        )
        self._cognition_graph = CognitionGraph(self.llm_adapter)

    def set_memory_context(self, memories: Any) -> None:
        self.memory_context = coerce_memory_package(memories)
        self._fallback_agent.set_memory_context(self.memory_context)

    def choose_action(self, observation: Observation):
        active_plan = self.current_plan if self.planning_enabled and self._plan_is_still_valid(observation) else ShortTermPlan()
        self.current_plan = active_plan
        self._fallback_agent.set_plan_context(active_plan)
        fallback_action = self._fallback_agent.choose_action(observation)
        planned_fallback = self._merge_action_with_plan(fallback_action, active_plan, observation.turn)
        action = self._cognition_graph.run(
            agent_id=self.agent_id,
            family_id=self.family_id,
            trait=self.trait.value,
            observation=observation,
            memory_context=self.memory_context,
            current_plan=active_plan if self.planning_enabled else None,
            policy_bias={action_type.value: weight for action_type, weight in self.policy_bias.action_weights.items()},
            fallback_action=planned_fallback,
        )
        self.current_plan = self._extract_plan(action, observation.turn, active_plan) if self.planning_enabled else ShortTermPlan()
        self._fallback_agent.set_plan_context(self.current_plan)
        return action

    def _plan_is_still_valid(self, observation: Observation) -> bool:
        if self.current_plan.is_empty():
            return False
        if observation.turn - self.current_plan.created_turn >= 4:
            return False
        if observation.self_state.energy <= 2 and any(
            token in self.current_plan.current_goal.lower()
            for token in ("explore", "cooperate", "broadcast", "message")
        ):
            return False
        if self.current_plan.planned_target_position is not None:
            target_x, target_y = self.current_plan.planned_target_position
            if (observation.self_state.position.x, observation.self_state.position.y) == (target_x, target_y):
                return False
        return True

    def _extract_plan(self, action: AgentAction, turn: int, previous_plan: ShortTermPlan) -> ShortTermPlan:
        current_goal = action.current_goal.strip() or previous_plan.current_goal
        planned_target_position = action.planned_target_position
        if planned_target_position is None and action.action_type is ActionType.MOVE and action.direction is not None:
            planned_target_position = previous_plan.planned_target_position
        if not current_goal and planned_target_position is None:
            return ShortTermPlan()
        created_turn = previous_plan.created_turn if current_goal == previous_plan.current_goal and previous_plan.created_turn >= 0 else turn
        return ShortTermPlan(
            current_goal=current_goal,
            planned_target_position=planned_target_position,
            created_turn=created_turn,
        )

    def _merge_action_with_plan(self, action: AgentAction, plan: ShortTermPlan, turn: int) -> AgentAction:
        if plan.is_empty():
            return action
        created_turn = plan.created_turn if plan.created_turn >= 0 else turn
        preserved_plan = ShortTermPlan(plan.current_goal, plan.planned_target_position, created_turn)
        return AgentAction(
            action_type=action.action_type,
            direction=action.direction,
            content=action.content,
            message_intent=action.message_intent,
            message_target=action.message_target,
            resource_hint=action.resource_hint,
            share_with_nearby=action.share_with_nearby,
            decision_source=action.decision_source,
            decision_note=action.decision_note,
            current_goal=preserved_plan.current_goal,
            planned_target_position=preserved_plan.planned_target_position,
        )