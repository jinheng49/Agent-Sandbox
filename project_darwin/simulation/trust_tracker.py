from __future__ import annotations

from dataclasses import dataclass

from project_darwin.agents.action_space import MessageIntent
from project_darwin.environment.observation_builder import SocialHint
from project_darwin.simulation.event_bus import EventBus, EventType, SimulationEvent
from project_darwin.simulation.run_context import SimulationConfig
from project_darwin.simulation.state import Position, WorldState


@dataclass(slots=True)
class PendingSignal:
    sender_id: str
    sender_family_id: str
    receiver_id: str
    turn: int
    target_position: Position
    resource_hint: str | None
    intent: str


@dataclass(frozen=True, slots=True)
class SocialAssessment:
    sender_reputation: float
    message_utility: float
    alliance_likelihood: float
    threat_level: float


class TrustTracker:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.pending_signals: list[PendingSignal] = []
        self.scores: dict[str, dict[str, float]] = {}
        self._processed_event_count = 0

    def process_events(self, event_bus: EventBus, world: WorldState) -> None:
        if not self.config.trust_enabled:
            return

        new_events = list(event_bus.events[self._processed_event_count :])
        for event in new_events:
            if event.event_type is EventType.MESSAGE:
                self._register_message(event, world)
            elif event.event_type in {EventType.FORAGE, EventType.FORAGE_MISS}:
                self._resolve_signal(event, event_bus, world)

        self._expire_stale_signals(world.turn)
        self._processed_event_count = len(event_bus.events)

    def get_hints(self, agent_id: str, receiver_family_id: str, current_turn: int) -> list[SocialHint]:
        hints: list[SocialHint] = []
        for signal in self.pending_signals:
            if signal.receiver_id != agent_id:
                continue
            if current_turn - signal.turn > self.config.trust_window:
                continue
            assessment = self.assess_signal(
                agent_id,
                receiver_family_id,
                signal.sender_id,
                signal.sender_family_id,
                signal.intent,
            )
            hints.append(
                SocialHint(
                    sender_id=signal.sender_id,
                    sender_family_id=signal.sender_family_id,
                    intent=signal.intent,
                    resource_hint=signal.resource_hint,
                    target_position=signal.target_position,
                    trust_score=self.scores.get(agent_id, {}).get(signal.sender_id, 0.0),
                    sender_reputation=assessment.sender_reputation,
                    message_utility=assessment.message_utility,
                    alliance_likelihood=assessment.alliance_likelihood,
                    threat_level=assessment.threat_level,
                )
            )
        hints.sort(
            key=lambda hint: (
                hint.message_utility,
                hint.sender_reputation,
                hint.alliance_likelihood,
                -hint.threat_level,
                0 if hint.target_position is None else 1,
            ),
            reverse=True,
        )
        return hints

    def get_score(self, receiver_id: str, sender_id: str) -> float:
        return self.scores.get(receiver_id, {}).get(sender_id, 0.0)

    def assess_signal(
        self,
        receiver_id: str,
        receiver_family_id: str,
        sender_id: str,
        sender_family_id: str,
        intent: str,
    ) -> SocialAssessment:
        trust_score = self.get_score(receiver_id, sender_id)
        sender_scores = self.scores.get(receiver_id, {})
        known_scores = list(sender_scores.values())
        sender_reputation = trust_score if sender_id in sender_scores else 0.0
        if known_scores:
            sender_reputation = max(-2.0, min(2.0, sender_reputation + (sum(known_scores) / len(known_scores)) * 0.15))

        alliance_likelihood = sender_reputation * 0.45
        if sender_family_id == receiver_family_id:
            alliance_likelihood += 0.1
        if intent in {MessageIntent.SHARE_GOLD.value, MessageIntent.SHARE_FOOD.value}:
            alliance_likelihood += 0.35
        if intent == MessageIntent.CLAIM_GOLD.value:
            alliance_likelihood -= 0.15
        if intent == MessageIntent.FALSE_GOLD.value:
            alliance_likelihood -= 0.45

        threat_level = 0.2
        if intent in {MessageIntent.CLAIM_GOLD.value, MessageIntent.FALSE_GOLD.value}:
            threat_level += 0.45
        if sender_reputation < 0:
            threat_level += min(abs(sender_reputation) * 0.3, 0.5)

        message_utility = sender_reputation * 0.4 + alliance_likelihood * 0.35 - threat_level * 0.25
        if intent in {MessageIntent.SHARE_GOLD.value, MessageIntent.SHARE_FOOD.value}:
            message_utility += 0.25
        if intent == MessageIntent.FALSE_GOLD.value:
            message_utility -= 0.55

        return SocialAssessment(
            sender_reputation=round(max(-2.0, min(2.0, sender_reputation)), 3),
            message_utility=round(max(-2.0, min(2.0, message_utility)), 3),
            alliance_likelihood=round(max(-1.0, min(1.0, alliance_likelihood)), 3),
            threat_level=round(max(0.0, min(2.0, threat_level)), 3),
        )

    def serialize_scores(self) -> dict[str, dict[str, float]]:
        return {
            receiver_id: {sender_id: round(score, 3) for sender_id, score in sorted(sender_scores.items())}
            for receiver_id, sender_scores in sorted(self.scores.items())
        }

    def _register_message(self, event: SimulationEvent, world: WorldState) -> None:
        message = event.payload.get("message", {})
        target = message.get("target_position")
        if event.agent_id is None or event.family_id is None or target is None:
            return

        intent = str(message.get("intent", ""))
        if intent not in {
            MessageIntent.SHARE_FOOD.value,
            MessageIntent.SHARE_GOLD.value,
            MessageIntent.CLAIM_GOLD.value,
            MessageIntent.FALSE_GOLD.value,
        }:
            return

        target_position = Position(int(target["x"]), int(target["y"]))
        resource_hint = message.get("resource_hint")

        self.pending_signals = [
            signal
            for signal in self.pending_signals
            if not (
                signal.sender_id == event.agent_id
                and signal.turn == event.turn
                and signal.target_position == target_position
            )
        ]

        for receiver_id, receiver in world.agents.items():
            if receiver_id == event.agent_id or not receiver.alive:
                continue
            self.pending_signals.append(
                PendingSignal(
                    sender_id=event.agent_id,
                    sender_family_id=event.family_id,
                    receiver_id=receiver_id,
                    turn=event.turn,
                    target_position=target_position,
                    resource_hint=str(resource_hint) if resource_hint is not None else None,
                    intent=intent,
                )
            )

    def _resolve_signal(self, event: SimulationEvent, event_bus: EventBus, world: WorldState) -> None:
        if event.agent_id is None:
            return

        position_payload = event.payload.get("position")
        if position_payload is None:
            return
        position = Position(int(position_payload["x"]), int(position_payload["y"]))
        resolved_resource = str(event.payload.get("resource")) if event.event_type is EventType.FORAGE else None

        remaining_signals: list[PendingSignal] = []
        for signal in self.pending_signals:
            if signal.receiver_id != event.agent_id or signal.target_position != position:
                remaining_signals.append(signal)
                continue
            if event.turn - signal.turn > self.config.trust_window:
                continue

            if event.event_type is EventType.FORAGE and resolved_resource == signal.resource_hint:
                self._update_score(
                    receiver_id=signal.receiver_id,
                    sender_id=signal.sender_id,
                    delta=self.config.trust_reward,
                    reason="validated_hint",
                    turn=event.turn,
                    event_bus=event_bus,
                    world=world,
                )
                continue

            penalty = self.config.trust_penalty + (0.4 if signal.intent == MessageIntent.FALSE_GOLD.value else 0.0)
            self._update_score(
                receiver_id=signal.receiver_id,
                sender_id=signal.sender_id,
                delta=-penalty,
                reason="misleading_hint",
                turn=event.turn,
                event_bus=event_bus,
                world=world,
            )

        self.pending_signals = remaining_signals

    def _update_score(
        self,
        *,
        receiver_id: str,
        sender_id: str,
        delta: float,
        reason: str,
        turn: int,
        event_bus: EventBus,
        world: WorldState,
    ) -> None:
        sender_scores = self.scores.setdefault(receiver_id, {})
        current_score = sender_scores.get(sender_id, 0.0)
        new_score = max(-2.0, min(2.0, current_score + delta))
        sender_scores[sender_id] = new_score
        event_bus.record(
            turn=turn,
            event_type=EventType.TRUST_UPDATE,
            agent_id=receiver_id,
            family_id=world.agents[receiver_id].family_id,
            payload={
                "sender_id": sender_id,
                "delta": round(delta, 3),
                "score": round(new_score, 3),
                "reason": reason,
            },
        )

    def _expire_stale_signals(self, current_turn: int) -> None:
        self.pending_signals = [
            signal for signal in self.pending_signals if current_turn - signal.turn <= self.config.trust_window
        ]