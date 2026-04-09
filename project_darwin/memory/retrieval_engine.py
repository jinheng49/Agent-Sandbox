from dataclasses import dataclass, field
from typing import Any

from project_darwin.environment.observation_builder import Observation
from project_darwin.memory.lineage_store import LineageStore, RetrievedMemory
from project_darwin.simulation.state import ResourceType


MEMORY_TYPE_PRIORITY = (
    "death_reflection",
    "success_reflection",
    "cooperation_reflection",
    "deception_reflection",
)


@dataclass(frozen=True, slots=True)
class MemoryDirective:
    memory_type: str
    priority: float
    lesson: str
    tags: list[str] = field(default_factory=list)
    hard_constraint: str | None = None
    soft_hint: str | None = None
    example: str | None = None
    action_bias: dict[str, float] = field(default_factory=dict)
    target_preference: str | None = None
    caution_against: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryContextPackage:
    directives: list[MemoryDirective] = field(default_factory=list)
    hard_constraints: list[str] = field(default_factory=list)
    soft_hints: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    typed_lessons: list[str] = field(default_factory=list)


def coerce_memory_package(memories: Any) -> MemoryContextPackage:
    if isinstance(memories, MemoryContextPackage):
        return memories
    if not memories:
        return MemoryContextPackage()
    if isinstance(memories, list) and all(isinstance(memory, str) for memory in memories):
        return MemoryContextPackage(soft_hints=list(memories), typed_lessons=list(memories))
    raise TypeError(f"Unsupported memory context type: {type(memories)!r}")


class RetrievalEngine:
    def __init__(self, lineage_store: LineageStore) -> None:
        self.lineage_store = lineage_store

    def get_relevant_memories(self, observation: Observation, family_id: str, lineage_id: str) -> list[RetrievedMemory]:
        query_text = self._build_query_text(observation)
        ranked_memories: list[tuple[float, RetrievedMemory]] = []
        for memory_type in self._select_memory_types(observation):
            for memory in self.lineage_store.retrieve(
                family_id=family_id,
                lineage_id=lineage_id,
                query_text=query_text,
                memory_type=memory_type,
            ):
                ranked_memories.append((self._memory_priority(memory, observation), memory))

        if not ranked_memories:
            return []

        deduplicated: dict[tuple[str, str, str], tuple[float, RetrievedMemory]] = {}
        for priority, memory in ranked_memories:
            key = (
                str(memory.metadata.get("memory_type", "unknown")),
                str(memory.metadata.get("source_run_id", "")),
                memory.lesson,
            )
            existing = deduplicated.get(key)
            if existing is None or priority > existing[0]:
                deduplicated[key] = (priority, memory)

        ordered = sorted(deduplicated.values(), key=lambda item: item[0], reverse=True)
        return [memory for _priority, memory in ordered[: self.lineage_store.config.memory_limit]]

    def get_memory_package(self, observation: Observation, family_id: str, lineage_id: str) -> MemoryContextPackage:
        memories = self.get_relevant_memories(observation, family_id, lineage_id)
        directives = [self._memory_directive(memory) for memory in memories]
        return MemoryContextPackage(
            directives=directives,
            hard_constraints=[directive.hard_constraint for directive in directives if directive.hard_constraint][:3],
            soft_hints=[directive.soft_hint for directive in directives if directive.soft_hint][:4],
            examples=[directive.example for directive in directives if directive.example][:3],
            typed_lessons=[self._format_memory_lesson(memory) for memory in memories],
        )

    def get_memory_lessons(self, observation: Observation, family_id: str, lineage_id: str) -> list[str]:
        return self.get_memory_package(observation, family_id, lineage_id).typed_lessons

    def _select_memory_types(self, observation: Observation) -> list[str]:
        selected: list[str] = []
        if observation.self_state.energy <= 3:
            selected.append("death_reflection")
        else:
            selected.append("success_reflection")
        if observation.nearby_agents or observation.social_hints:
            selected.append("cooperation_reflection")
        if observation.recent_received_messages or any(hint.trust_score < 0 for hint in observation.social_hints):
            selected.append("deception_reflection")

        for memory_type in MEMORY_TYPE_PRIORITY:
            if memory_type not in selected:
                selected.append(memory_type)
        return selected

    def _memory_priority(self, memory: RetrievedMemory, observation: Observation) -> float:
        memory_type = str(memory.metadata.get("memory_type", "unknown"))
        priority = memory.score
        if memory_type == "death_reflection" and observation.self_state.energy <= 3:
            priority += 0.35
        if memory_type == "success_reflection" and observation.self_state.energy > 3:
            priority += 0.2
        if memory_type == "cooperation_reflection" and observation.nearby_agents:
            priority += 0.25
        if memory_type == "deception_reflection" and observation.recent_received_messages:
            priority += 0.3
        if any(tag == "false_gold" for tag in memory.metadata.get("tags", [])):
            priority += 0.05
        return round(priority, 6)

    def _format_memory_lesson(self, memory: RetrievedMemory) -> str:
        memory_type = str(memory.metadata.get("memory_type", "unknown")).replace("_reflection", "")
        return f"[{memory_type}] {memory.lesson}"

    def _memory_directive(self, memory: RetrievedMemory) -> MemoryDirective:
        memory_type = str(memory.metadata.get("memory_type", "unknown"))
        tags = [str(tag) for tag in memory.metadata.get("tags", [])]
        hard_constraint: str | None = None
        soft_hint: str | None = None
        example: str | None = f"When {memory.situation}, strategy was: {memory.lesson}"
        action_bias: dict[str, float] = {}
        target_preference: str | None = None
        caution_against: str | None = None

        if memory_type == "death_reflection":
            hard_constraint = "Avoid high-cost messaging and detours when energy is critically low."
            soft_hint = memory.lesson
            action_bias = {"message": -1.0, "rest": 0.3, "forage": 0.5}
            target_preference = "seek_resource"
        elif memory_type == "success_reflection":
            soft_hint = memory.lesson
            action_bias = {"move": 0.2, "forage": 0.4}
            target_preference = "seek_resource"
        elif memory_type == "cooperation_reflection":
            hard_constraint = "If allies are adjacent near gold, prefer cooperative collection over solo wandering."
            soft_hint = memory.lesson
            action_bias = {"message": 0.5, "forage": 0.6}
            target_preference = "cooperate_on_gold"
        elif memory_type == "deception_reflection":
            hard_constraint = "Do not reroute solely on unverified suspicious signals."
            soft_hint = memory.lesson
            action_bias = {"move": 0.2, "message": -0.4}
            target_preference = "seek_verified_resource"
            caution_against = "suspicious_signal"

        if "false_gold" in tags:
            caution_against = "false_gold"

        return MemoryDirective(
            memory_type=memory_type,
            priority=memory.score,
            lesson=memory.lesson,
            tags=tags,
            hard_constraint=hard_constraint,
            soft_hint=soft_hint,
            example=example,
            action_bias=action_bias,
            target_preference=target_preference,
            caution_against=caution_against,
        )

    def _build_query_text(self, observation: Observation) -> str:
        visible_gold = sum(
            1 for resource in observation.visible_resources.values() if resource.kind is ResourceType.GOLD
        )
        visible_food = sum(
            1 for resource in observation.visible_resources.values() if resource.kind is ResourceType.FOOD
        )
        return (
            f"turn {observation.turn} "
            f"energy {observation.self_state.energy} "
            f"gold_visible {visible_gold} "
            f"food_visible {visible_food} "
            f"nearby_agents {len(observation.nearby_agents)} "
            f"recent_messages {len(observation.recent_received_messages)} "
            f"unique_positions {observation.unique_positions_visited} "
            f"exploration_ratio {round(observation.exploration_ratio, 4)} "
            f"hotspots {' '.join(hotspot.resource_kind for hotspot in observation.resource_hotspots[:3])} "
            f"recent_events {' '.join(event.event_type for event in observation.recent_self_events[-4:])}"
        )
