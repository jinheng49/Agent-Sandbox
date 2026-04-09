from project_darwin.memory.lineage_store import MemoryRecord
from project_darwin.simulation.event_bus import EventBus, EventType
from project_darwin.simulation.state import WorldState


class ReflectionEngine:
    def summarize_run_memories(
        self,
        event_bus: EventBus,
        world: WorldState,
        *,
        experiment_id: str,
        run_group: str,
        lineage_id: str,
        generation: int,
        run_id: str,
        reflection_window: int,
        agent_traits: dict[str, str],
    ) -> list[MemoryRecord]:
        reflections = self.summarize_deaths(
            event_bus,
            experiment_id=experiment_id,
            run_group=run_group,
            lineage_id=lineage_id,
            generation=generation,
            run_id=run_id,
            reflection_window=reflection_window,
            agent_traits=agent_traits,
        )
        reflections.extend(
            self._summarize_survivors(
                event_bus,
                world,
                experiment_id=experiment_id,
                run_group=run_group,
                lineage_id=lineage_id,
                generation=generation,
                run_id=run_id,
                reflection_window=reflection_window,
                agent_traits=agent_traits,
            )
        )
        reflections.extend(
            self._summarize_cooperation(
                event_bus,
                world,
                experiment_id=experiment_id,
                run_group=run_group,
                lineage_id=lineage_id,
                generation=generation,
                run_id=run_id,
                agent_traits=agent_traits,
            )
        )
        reflections.extend(
            self._summarize_deception(
                event_bus,
                world,
                experiment_id=experiment_id,
                run_group=run_group,
                lineage_id=lineage_id,
                generation=generation,
                run_id=run_id,
                reflection_window=reflection_window,
                agent_traits=agent_traits,
            )
        )
        return reflections

    def summarize_deaths(
        self,
        event_bus: EventBus,
        *,
        experiment_id: str,
        run_group: str,
        lineage_id: str,
        generation: int,
        run_id: str,
        reflection_window: int,
        agent_traits: dict[str, str],
    ) -> list[MemoryRecord]:
        reflections: list[MemoryRecord] = []
        for event in event_bus.events:
            if event.event_type is not EventType.DEATH or event.family_id is None or event.agent_id is None:
                continue

            recent_events = [
                candidate
                for candidate in event_bus.events
                if candidate.agent_id == event.agent_id and candidate.turn <= event.turn
            ][-reflection_window:]

            move_count = sum(1 for candidate in recent_events if candidate.event_type is EventType.MOVE)
            forage_count = sum(1 for candidate in recent_events if candidate.event_type is EventType.FORAGE)
            message_count = sum(1 for candidate in recent_events if candidate.event_type is EventType.MESSAGE)
            cooperation_count = sum(1 for candidate in recent_events if candidate.event_type is EventType.COOPERATION)

            tags = ["death", "energy_depletion"]
            if forage_count == 0:
                tags.append("missed_resources")
            if message_count >= 2:
                tags.append("overcommunicated")
            if cooperation_count > 0:
                tags.append("cooperate")

            lesson = self._build_lesson(tags)
            situation = (
                f"death_turn={event.turn}, moves={move_count}, forage={forage_count}, "
                f"messages={message_count}, cooperation={cooperation_count}"
            )

            reflections.append(
                MemoryRecord(
                    experiment_id=experiment_id,
                    run_group=run_group,
                    family_id=event.family_id,
                    lineage_id=lineage_id,
                    generation=generation,
                    trait=agent_traits.get(event.agent_id, "unknown"),
                    death_reason=str(event.payload.get("reason", "unknown")),
                    memory_type="death_reflection",
                    source_run_id=run_id,
                    source_agent_id=event.agent_id,
                    death_turn=event.turn,
                    situation=situation,
                    lesson=lesson,
                    tags=tags,
                )
            )

        return reflections

    def _summarize_survivors(
        self,
        event_bus: EventBus,
        world: WorldState,
        *,
        experiment_id: str,
        run_group: str,
        lineage_id: str,
        generation: int,
        run_id: str,
        reflection_window: int,
        agent_traits: dict[str, str],
    ) -> list[MemoryRecord]:
        reflections: list[MemoryRecord] = []
        for agent in world.agents.values():
            if not agent.alive:
                continue

            recent_events = [
                candidate
                for candidate in event_bus.events
                if candidate.agent_id == agent.agent_id and candidate.turn <= world.turn
            ][-reflection_window:]

            move_count = sum(1 for candidate in recent_events if candidate.event_type is EventType.MOVE)
            forage_count = sum(1 for candidate in recent_events if candidate.event_type is EventType.FORAGE)
            message_count = sum(1 for candidate in recent_events if candidate.event_type is EventType.MESSAGE)
            cooperation_count = sum(1 for candidate in recent_events if candidate.event_type is EventType.COOPERATION)

            tags = ["survival", "stable_energy"]
            if forage_count > 0:
                tags.append("resourceful")
            if cooperation_count > 0:
                tags.append("cooperate")
            if message_count == 0:
                tags.append("signal_efficient")

            lesson = self._build_survival_lesson(tags)
            situation = (
                f"survived_until={world.turn}, moves={move_count}, forage={forage_count}, "
                f"messages={message_count}, cooperation={cooperation_count}, energy={agent.energy}"
            )
            reflections.append(
                MemoryRecord(
                    experiment_id=experiment_id,
                    run_group=run_group,
                    family_id=agent.family_id,
                    lineage_id=lineage_id,
                    generation=generation,
                    trait=agent_traits.get(agent.agent_id, "unknown"),
                    death_reason="survived",
                    memory_type="success_reflection",
                    source_run_id=run_id,
                    source_agent_id=agent.agent_id,
                    death_turn=world.turn,
                    situation=situation,
                    lesson=lesson,
                    tags=tags,
                )
            )
        return reflections

    def _summarize_deception(
        self,
        event_bus: EventBus,
        world: WorldState,
        *,
        experiment_id: str,
        run_group: str,
        lineage_id: str,
        generation: int,
        run_id: str,
        reflection_window: int,
        agent_traits: dict[str, str],
    ) -> list[MemoryRecord]:
        reflections: list[MemoryRecord] = []
        message_events = [event for event in event_bus.events if event.event_type is EventType.MESSAGE]
        for event in event_bus.events:
            if event.event_type is not EventType.TRUST_UPDATE or event.agent_id is None or event.family_id is None:
                continue
            if str(event.payload.get("reason", "")) != "misleading_hint":
                continue

            sender_id = str(event.payload.get("sender_id", ""))
            if not sender_id:
                continue
            sender_message = self._latest_sender_message_before_turn(message_events, sender_id, event.turn)
            if sender_message is None:
                continue

            recent_events = [
                candidate
                for candidate in event_bus.events
                if candidate.agent_id == event.agent_id and candidate.turn <= event.turn
            ][-reflection_window:]
            distrust_count = sum(
                1
                for candidate in recent_events
                if candidate.event_type is EventType.TRUST_UPDATE
                and str(candidate.payload.get("reason", "")) == "misleading_hint"
            )
            message = sender_message.payload.get("message", {}) if isinstance(sender_message.payload.get("message"), dict) else {}
            intent = str(message.get("intent", "contact"))
            target = message.get("target_position", {}) if isinstance(message.get("target_position"), dict) else {}
            sender_family_id = str(message.get("sender_family_id", sender_message.family_id or "unknown"))
            tags = ["deception", "misleading_hint", intent]
            if intent == "false_gold":
                tags.append("false_gold")
            if distrust_count > 1:
                tags.append("repeat_pattern")

            reflections.append(
                MemoryRecord(
                    experiment_id=experiment_id,
                    run_group=run_group,
                    family_id=event.family_id,
                    lineage_id=lineage_id,
                    generation=generation,
                    trait=agent_traits.get(event.agent_id, "unknown"),
                    death_reason="misleading_hint",
                    memory_type="deception_reflection",
                    source_run_id=run_id,
                    source_agent_id=event.agent_id,
                    death_turn=event.turn,
                    situation=(
                        f"receiver={event.agent_id}, sender={sender_id}, sender_family={sender_family_id}, "
                        f"intent={intent}, target=({target.get('x', 0)},{target.get('y', 0)}), distrust_count={distrust_count}"
                    ),
                    lesson=self._build_deception_lesson(intent, sender_family_id, tags),
                    tags=tags,
                )
            )
        return reflections

    def _latest_sender_message_before_turn(self, message_events, sender_id: str, turn: int):
        candidates = [
            event
            for event in message_events
            if event.agent_id == sender_id and event.turn <= turn
        ]
        if not candidates:
            return None
        return candidates[-1]

    def _summarize_cooperation(
        self,
        event_bus: EventBus,
        world: WorldState,
        *,
        experiment_id: str,
        run_group: str,
        lineage_id: str,
        generation: int,
        run_id: str,
        agent_traits: dict[str, str],
    ) -> list[MemoryRecord]:
        reflections: list[MemoryRecord] = []
        for event in event_bus.events:
            if event.event_type is not EventType.COOPERATION or event.agent_id is None:
                continue

            participant_ids = [event.agent_id, *list(event.payload.get("participant_ids", []))]
            position = event.payload.get("position", {})
            for participant_id in participant_ids:
                participant = world.agents.get(participant_id)
                if participant is None:
                    continue
                reflections.append(
                    MemoryRecord(
                        experiment_id=experiment_id,
                        run_group=run_group,
                        family_id=participant.family_id,
                        lineage_id=lineage_id,
                        generation=generation,
                        trait=agent_traits.get(participant_id, "unknown"),
                        death_reason="cooperation_success",
                        memory_type="cooperation_reflection",
                        source_run_id=run_id,
                        source_agent_id=participant_id,
                        death_turn=event.turn,
                        situation=(
                            f"cooperation_turn={event.turn}, kind={event.payload.get('kind', 'unknown')}, "
                            f"participants={len(participant_ids)}, position=({position.get('x', 0)},{position.get('y', 0)})"
                        ),
                        lesson="Cooperate near gold when allies are adjacent; shared gains can preserve multiple agents.",
                        tags=["cooperate", "gold_share", "shared_gain"],
                    )
                )
        return reflections

    def _build_lesson(self, tags: list[str]) -> str:
        if "overcommunicated" in tags:
            return "Reduce broadcasts when low on energy; communication cost can outweigh survival gains."
        if "missed_resources" in tags:
            return "Seek nearby resources earlier instead of prolonged wandering."
        if "cooperate" in tags:
            return "Cooperate near gold when possible; shared gains can offset scarce energy."
        return "Maintain energy discipline and prioritize reachable resources before exploring further."

    def _build_survival_lesson(self, tags: list[str]) -> str:
        if "cooperate" in tags:
            return "Repeat cooperative collection when nearby allies can share gains without overextending."
        if "signal_efficient" in tags:
            return "Prioritize nearby resources and avoid unnecessary broadcasts while energy is stable."
        return "Maintain balanced exploration and keep moving toward reachable resources that sustain survival."

    def _build_deception_lesson(self, intent: str, sender_family_id: str, tags: list[str]) -> str:
        if "false_gold" in tags:
            return (
                f"Treat repeated {intent} signals from family {sender_family_id} as suspicious until they are validated by nearby evidence."
            )
        return (
            f"Reduce route changes caused by unverified signals from family {sender_family_id}; wait for corroboration or local evidence first."
        )
