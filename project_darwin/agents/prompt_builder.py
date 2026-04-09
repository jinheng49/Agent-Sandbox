import json
from typing import Any

from project_darwin.agents.action_space import AgentAction, ActionType, ShortTermPlan, action_schema_text
from project_darwin.agents.policy import PolicyBias, apply_policy_bias, extract_features, get_action_scores
from project_darwin.agents.traits import TraitProfile, get_trait_config
from project_darwin.environment.observation_builder import Observation
from project_darwin.memory.retrieval_engine import MemoryContextPackage, coerce_memory_package


def build_llm_prompts(
    *,
    agent_id: str,
    family_id: str,
    trait: str,
    observation: Observation,
    memory_context: MemoryContextPackage | list[str],
    current_plan: ShortTermPlan | None,
    policy_bias: dict[str, float],
    heuristic_recommendation: AgentAction | None,
) -> tuple[str, str]:
    memory_package = coerce_memory_package(memory_context)
    system_prompt = (
        "You are deciding a single action for an agent in Project Darwin. "
        "Return exactly one JSON object and nothing else. Do not use markdown. "
        "The JSON must satisfy the required schema.\n\n"
        "Decision priorities:\n"
        "1. Preserve survival under energy pressure.\n"
        "2. Stay consistent with the agent trait and instinct bias.\n"
        "3. Use family memory when it is relevant to the current situation.\n"
        "4. Keep a short-term goal for the next 2 to 4 turns when it still makes sense.\n"
        "5. Respect visible resources, nearby agents, and trust hints.\n\n"
        "You will receive a neural-symbolic fusion block that contains: "
        "(a) instinct scores from the fast heuristic layer, "
        "(b) family memory lessons retrieved from lineage storage, and "
        "(c) a heuristic draft action. "
        "Treat those as priors, then produce the final action JSON.\n\n"
        "Always include current_goal and planned_target_position in the JSON. "
        "If an existing plan is still valid, continue it instead of inventing a new one.\n\n"
        "Required JSON schema:\n"
        f"{action_schema_text()}"
    )
    user_prompt = json.dumps(
        _build_prompt_context(
            agent_id=agent_id,
            family_id=family_id,
            trait=trait,
            observation=observation,
            memory_context=memory_package,
            current_plan=current_plan,
            policy_bias=policy_bias,
            heuristic_recommendation=heuristic_recommendation,
        ),
        indent=2,
        ensure_ascii=True,
    )
    return system_prompt, user_prompt


def build_repair_prompt(raw_output: str, error_message: str) -> str:
    return json.dumps(
        {
            "instruction": "Repair the invalid model output and return one valid JSON object only.",
            "error": error_message,
            "previous_output": raw_output,
        },
        indent=2,
        ensure_ascii=True,
    )


def _build_prompt_context(
    *,
    agent_id: str,
    family_id: str,
    trait: str,
    observation: Observation,
    memory_context: MemoryContextPackage,
    current_plan: ShortTermPlan | None,
    policy_bias: dict[str, float],
    heuristic_recommendation: AgentAction | None,
) -> dict[str, Any]:
    trait_profile = TraitProfile(trait)
    trait_config = get_trait_config(trait_profile)
    base_scores = get_action_scores(observation, trait_config)
    merged_scores = apply_policy_bias(base_scores, _policy_bias_from_dict(policy_bias))
    features = extract_features(observation)
    instinct_summary = _summarize_instinct(merged_scores)
    memory_summary = _summarize_memory(memory_context)
    heuristic_summary = _summarize_heuristic_recommendation(heuristic_recommendation)
    return {
        "agent": {
            "agent_id": agent_id,
            "family_id": family_id,
            "trait": trait,
        },
        "planning": {
            "current_plan": {
                "goal": "" if current_plan is None else current_plan.current_goal,
                "planned_target_position": None
                if current_plan is None or current_plan.planned_target_position is None
                else {"x": current_plan.planned_target_position[0], "y": current_plan.planned_target_position[1]},
                "created_turn": -1 if current_plan is None else current_plan.created_turn,
            },
            "instruction": "Return a short-term goal, an optional target position, and the best action for this turn.",
        },
        "instinct": {
            "summary": instinct_summary,
            "features": {
                "energy": features.energy,
                "energy_low": features.energy_low,
                "on_resource": features.on_resource,
                "visible_food_count": features.visible_food_count,
                "visible_gold_count": features.visible_gold_count,
                "nearby_agent_count": features.nearby_agent_count,
                "recent_message_count": features.recent_message_count,
                "known_hotspot_count": features.known_hotspot_count,
                "exploration_ratio": round(features.exploration_ratio, 4),
            },
            "action_scores": {action_type.value: round(score, 3) for action_type, score in merged_scores.items()},
        },
        "neural_symbolic_fusion": {
            "instruction": (
                "Fuse the fast instinct prior, the retrieved family memories, and the heuristic draft action. "
                "Override them only when the current observation gives stronger evidence."
            ),
            "instinct_summary": instinct_summary,
            "memory_summary": memory_summary,
            "heuristic_recommendation": heuristic_summary,
        },
        "observation": {
            "turn": observation.turn,
            "self": {
                "position": {"x": observation.self_state.position.x, "y": observation.self_state.position.y},
                "energy": observation.self_state.energy,
                "inventory": dict(observation.self_state.inventory),
            },
            "history": {
                "recent_positions": [
                    {"x": position.x, "y": position.y}
                    for position in observation.recent_positions[-6:]
                ],
                "explored_positions": [
                    {"x": position.x, "y": position.y}
                    for position in observation.explored_positions[-12:]
                ],
                "nearby_unexplored_positions": [
                    {"x": position.x, "y": position.y}
                    for position in observation.nearby_unexplored_positions[:8]
                ],
                "unique_positions_visited": observation.unique_positions_visited,
                "exploration_ratio": round(observation.exploration_ratio, 4),
                "recent_events": [
                    {
                        "turn": event.turn,
                        "event_type": event.event_type,
                        "position": None
                        if event.position is None
                        else {"x": event.position.x, "y": event.position.y},
                        "detail": event.detail,
                    }
                    for event in observation.recent_self_events[-6:]
                ],
                "recent_received_messages": [
                    {
                        "turn": message.turn,
                        "sender_id": message.sender_id,
                        "sender_family_id": message.sender_family_id,
                        "content": message.content,
                        "intent": message.intent,
                        "resource_hint": message.resource_hint,
                        "target_position": None
                        if message.target_position is None
                        else {"x": message.target_position.x, "y": message.target_position.y},
                        "trust_score": round(message.trust_score, 3),
                        "sender_reputation": round(message.sender_reputation, 3),
                        "message_utility": round(message.message_utility, 3),
                        "alliance_likelihood": round(message.alliance_likelihood, 3),
                        "threat_level": round(message.threat_level, 3),
                    }
                    for message in observation.recent_received_messages[-6:]
                ],
                "resource_hotspots": [
                    {
                        "position": {"x": hotspot.position.x, "y": hotspot.position.y},
                        "resource_kind": hotspot.resource_kind,
                        "sightings": hotspot.sightings,
                        "last_seen_turn": hotspot.last_seen_turn,
                    }
                    for hotspot in observation.resource_hotspots[:6]
                ],
            },
            "visible_resources": [
                {
                    "x": position.x,
                    "y": position.y,
                    "kind": resource.kind.value,
                    "amount": resource.amount,
                }
                for position, resource in observation.visible_resources.items()
            ],
            "nearby_agents": [
                {
                    "agent_id": agent.agent_id,
                    "family_id": agent.family_id,
                    "position": {"x": agent.position.x, "y": agent.position.y},
                    "energy": agent.energy,
                }
                for agent in observation.nearby_agents
            ],
            "social_hints": [
                {
                    "sender_id": hint.sender_id,
                    "sender_family_id": hint.sender_family_id,
                    "intent": hint.intent,
                    "resource_hint": hint.resource_hint,
                    "target_position": None
                    if hint.target_position is None
                    else {"x": hint.target_position.x, "y": hint.target_position.y},
                    "trust_score": round(hint.trust_score, 3),
                }
                for hint in observation.social_hints
            ],
            "agent_profiles": [
                {
                    "agent_id": profile.agent_id,
                    "family_id": profile.family_id,
                    "truth_score": round(profile.truth_score, 3),
                    "message_count": profile.message_count,
                    "false_gold_count": profile.false_gold_count,
                    "gold_competition_count": profile.gold_competition_count,
                    "sender_reputation": round(profile.sender_reputation, 3),
                    "alliance_likelihood": round(profile.alliance_likelihood, 3),
                    "threat_level": round(profile.threat_level, 3),
                }
                for profile in observation.agent_profiles
            ],
        },
        "family_memory": {
            "hard_constraints": memory_context.hard_constraints[:3],
            "soft_hints": memory_context.soft_hints[:4],
            "examples": memory_context.examples[:3],
            "typed_lessons": memory_context.typed_lessons[:5],
        },
        "valid_action_types": [action_type.value for action_type in ActionType],
    }


def _policy_bias_from_dict(policy_bias: dict[str, float]) -> PolicyBias:
    return PolicyBias(
        action_weights={
            action_type: float(policy_bias.get(action_type.value, 0.0))
            for action_type in ActionType
        }
    )


def _summarize_instinct(action_scores: dict[ActionType, float]) -> str:
    ranked_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    if not ranked_actions:
        return "Fast instinct layer produced no strong action preference."
    summary = ", ".join(f"{action_type.value} ({score:.2f})" for action_type, score in ranked_actions)
    return f"Fast instinct layer ranks actions as: {summary}."


def _summarize_memory(memory_context: MemoryContextPackage) -> str:
    snippets = [
        *memory_context.hard_constraints[:1],
        *memory_context.soft_hints[:1],
        *memory_context.examples[:1],
        *memory_context.typed_lessons[:1],
    ]
    if not snippets:
        return "No relevant family memories were retrieved for this turn."
    return f"Retrieved family memory suggests: {' | '.join(snippets)}"


def _summarize_heuristic_recommendation(heuristic_recommendation: AgentAction | None) -> str:
    if heuristic_recommendation is None:
        return "Heuristic layer did not produce a draft action."
    parts = [f"Heuristic draft action: {heuristic_recommendation.action_type.value}"]
    if heuristic_recommendation.direction is not None:
        parts.append(f"direction={heuristic_recommendation.direction.value}")
    if heuristic_recommendation.current_goal:
        parts.append(f"goal={heuristic_recommendation.current_goal}")
    if heuristic_recommendation.planned_target_position is not None:
        parts.append(f"target={heuristic_recommendation.planned_target_position}")
    return ", ".join(parts) + "."