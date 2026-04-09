from __future__ import annotations

import atexit
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from project_darwin.simulation.run_context import SimulationConfig


_CLIENT_CACHE: dict[str, QdrantClient] = {}


@dataclass(slots=True)
class MemoryRecord:
    experiment_id: str
    run_group: str
    family_id: str
    lineage_id: str
    generation: int
    trait: str
    death_reason: str
    memory_type: str
    source_run_id: str
    source_agent_id: str
    death_turn: int
    situation: str
    lesson: str
    tags: list[str]
    memory_id: str | None = None

    def as_payload(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id or str(uuid4()),
            "experiment_id": self.experiment_id,
            "run_group": self.run_group,
            "family_id": self.family_id,
            "lineage_id": self.lineage_id,
            "generation": self.generation,
            "trait": self.trait,
            "death_reason": self.death_reason,
            "memory_type": self.memory_type,
            "source_run_id": self.source_run_id,
            "source_agent_id": self.source_agent_id,
            "death_turn": self.death_turn,
            "situation": self.situation,
            "lesson": self.lesson,
            "tags": self.tags,
            "text": self.to_text(),
        }

    def to_text(self) -> str:
        return f"situation: {self.situation}. lesson: {self.lesson}. tags: {' '.join(self.tags)}"


@dataclass(slots=True)
class RetrievedMemory:
    text: str
    lesson: str
    situation: str
    generation: int
    score: float
    metadata: dict[str, object]


class LineageStore:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.collection_name = _build_collection_name(config)
        storage_path = config.artifact_dir / "qdrant" / self.collection_name
        storage_path.mkdir(parents=True, exist_ok=True)
        self.client = _get_or_create_client(storage_path)
        self._ensure_collection()

    def add_reflection(self, record: MemoryRecord) -> None:
        payload = record.as_payload()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=payload["memory_id"],
                    vector=_embed_text(str(payload["text"]), self.config.memory_vector_size),
                    payload=payload,
                )
            ],
        )

    def retrieve(
        self,
        family_id: str,
        lineage_id: str,
        query_text: str,
        *,
        memory_type: str | None = None,
    ) -> list[RetrievedMemory]:
        must_filters = [
            FieldCondition(key="family_id", match=MatchValue(value=family_id)),
            FieldCondition(key="lineage_id", match=MatchValue(value=lineage_id)),
        ]
        if memory_type is not None:
            must_filters.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=_embed_text(query_text, self.config.memory_vector_size),
            query_filter=Filter(must=must_filters),
            limit=self.config.memory_limit,
        ).points
        memories: list[RetrievedMemory] = []
        for result in search_results:
            if result.score < self.config.memory_score_threshold:
                continue
            payload = result.payload or {}
            metadata = {
                "experiment_id": payload.get("experiment_id", self.config.experiment_id),
                "run_group": payload.get("run_group", self.config.run_group),
                "family_id": payload.get("family_id", family_id),
                "lineage_id": payload.get("lineage_id", lineage_id),
                "trait": payload.get("trait", "unknown"),
                "death_reason": payload.get("death_reason", "unknown"),
                "memory_type": payload.get("memory_type", "unknown"),
                "tags": list(payload.get("tags", [])),
                "source_run_id": payload.get("source_run_id", ""),
                "source_agent_id": payload.get("source_agent_id", ""),
            }
            memories.append(
                RetrievedMemory(
                    text=str(payload.get("text", "")),
                    lesson=str(payload.get("lesson", "")),
                    situation=str(payload.get("situation", "")),
                    generation=int(payload.get("generation", 0)),
                    score=float(result.score),
                    metadata=metadata,
                )
            )
        return memories

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name, exact=True).count

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.config.memory_vector_size, distance=Distance.COSINE),
        )


def _embed_text(text: str, vector_size: int) -> list[float]:
    vector = [0.0] * vector_size
    tokens = [token for token in text.lower().replace(",", " ").replace(".", " ").split() if token]
    if not tokens:
        return vector

    for token in tokens:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        index = int(token_hash[:8], 16) % vector_size
        sign = 1.0 if int(token_hash[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _build_collection_name(config: SimulationConfig) -> str:
    return "_".join(
        [
            _slugify(config.memory_collection_name),
            _slugify(config.experiment_id),
            _slugify(config.run_group),
        ]
    )


def _slugify(value: str) -> str:
    sanitized = [character.lower() if character.isalnum() else "_" for character in value]
    collapsed = "".join(sanitized).strip("_")
    return collapsed or "default"


def _get_or_create_client(storage_path: Path) -> QdrantClient:
    cache_key = str(storage_path.resolve())
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = QdrantClient(path=cache_key)
    return _CLIENT_CACHE[cache_key]


def _close_cached_clients() -> None:
    for client in list(_CLIENT_CACHE.values()):
        try:
            client.close()
        except Exception:
            pass
    _CLIENT_CACHE.clear()


atexit.register(_close_cached_clients)
