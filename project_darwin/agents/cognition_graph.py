from dataclasses import replace
from dataclasses import dataclass

from project_darwin.agents.action_space import AgentAction, ShortTermPlan, parse_action_text
from project_darwin.agents.llm_adapter import LLMAdapter
from project_darwin.agents.prompt_builder import build_llm_prompts, build_repair_prompt
from project_darwin.environment.observation_builder import Observation
from project_darwin.memory.retrieval_engine import MemoryContextPackage, coerce_memory_package


@dataclass(slots=True)
class CognitionGraph:
    llm_adapter: LLMAdapter
    max_retries: int = 1

    def _model_label(self) -> str:
        return str(getattr(self.llm_adapter, "model_name", "configured_model"))

    def run(
        self,
        *,
        agent_id: str,
        family_id: str,
        trait: str,
        observation: Observation,
        memory_context: MemoryContextPackage | list[str],
        current_plan: ShortTermPlan | None,
        policy_bias: dict[str, float],
        fallback_action: AgentAction,
    ) -> AgentAction:
        memory_package = coerce_memory_package(memory_context)
        system_prompt, user_prompt = build_llm_prompts(
            agent_id=agent_id,
            family_id=family_id,
            trait=trait,
            observation=observation,
            memory_context=memory_package,
            current_plan=current_plan,
            policy_bias=policy_bias,
        )
        raw_output = ""
        fallback_reason = ""
        try:
            raw_output = self.llm_adapter.complete(system_prompt, user_prompt)
            return replace(
                parse_action_text(raw_output),
                decision_source="llm",
                decision_note=self._model_label(),
            )
        except Exception as first_error:
            fallback_reason = str(first_error)
            repair_prompt = build_repair_prompt(raw_output, str(first_error))
            for _ in range(self.max_retries):
                try:
                    repaired_output = self.llm_adapter.complete(system_prompt, repair_prompt)
                    return replace(
                        parse_action_text(repaired_output),
                        decision_source="llm_repair",
                        decision_note=self._model_label(),
                    )
                except Exception:
                    continue
        return replace(
            fallback_action,
            decision_source="heuristic_fallback",
            decision_note=fallback_reason[:160],
        )
