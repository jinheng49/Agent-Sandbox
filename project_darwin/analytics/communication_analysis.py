from collections import Counter
from dataclasses import dataclass
from math import log2
import zlib

from project_darwin.simulation.event_bus import EventBus, EventType


@dataclass(slots=True)
class CommunicationReport:
    vocabulary_size: int
    mean_message_length: float
    entropy: float
    word_frequency: dict[str, int]
    protocol_compression_rate: float
    deception_frequency: float
    share_gold_signals: int
    false_gold_signals: int


class CommunicationAnalysis:
    def _tokenize(self, message: str) -> list[str]:
        normalized = message.strip()
        if not normalized:
            return []
        if " " in normalized:
            return [token for token in normalized.split() if token]
        return list(normalized)

    def analyze(self, event_bus: EventBus) -> CommunicationReport:
        messages = [
            event.payload["message"]["content"]
            for event in event_bus.events
            if event.event_type is EventType.MESSAGE
        ]
        share_gold_signals = sum(
            1
            for event in event_bus.events
            if event.event_type is EventType.MESSAGE
            and event.payload["message"].get("intent") == "share_gold"
        )
        false_gold_signals = sum(
            1
            for event in event_bus.events
            if event.event_type is EventType.MESSAGE
            and event.payload["message"].get("intent") == "false_gold"
        )
        if not messages:
            return CommunicationReport(
                vocabulary_size=0,
                mean_message_length=0.0,
                entropy=0.0,
                word_frequency={},
                protocol_compression_rate=0.0,
                deception_frequency=0.0,
                share_gold_signals=share_gold_signals,
                false_gold_signals=false_gold_signals,
            )

        token_counter: Counter[str] = Counter()
        for message in messages:
            token_counter.update(self._tokenize(message))

        total_tokens = sum(token_counter.values())
        entropy = 0.0
        for count in token_counter.values():
            probability = count / total_tokens
            entropy -= probability * log2(probability)

        raw_stream = "\n".join(messages).encode("utf-8")
        compressed_stream = zlib.compress(raw_stream)
        compression_rate = len(compressed_stream) / max(len(raw_stream), 1)
        deception_frequency = false_gold_signals / max(len(messages), 1)

        return CommunicationReport(
            vocabulary_size=len(token_counter),
            mean_message_length=sum(len(message) for message in messages) / len(messages),
            entropy=entropy,
            word_frequency=dict(token_counter.most_common(20)),
            protocol_compression_rate=round(compression_rate, 4),
            deception_frequency=round(deception_frequency, 4),
            share_gold_signals=share_gold_signals,
            false_gold_signals=false_gold_signals,
        )
