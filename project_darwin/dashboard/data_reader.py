import json
from pathlib import Path
from typing import Any


def load_replay(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_replay_paths(root: Path | None = None) -> list[Path]:
    artifact_root = root or Path("artifacts")
    if not artifact_root.exists():
        return []

    replay_paths: list[Path] = []
    for candidate in sorted(artifact_root.rglob("*.json")):
        if candidate.name == "experiment_manifest.json":
            continue
        payload = load_replay(candidate)
        if {"metadata", "metrics", "events"}.issubset(payload):
            replay_paths.append(candidate)
    return replay_paths


def build_replay_catalog(root: Path | None = None) -> list[dict[str, Any]]:
    artifact_root = root or Path("artifacts")
    catalog: list[dict[str, Any]] = []
    for replay_path in discover_replay_paths(artifact_root):
        payload = load_replay(replay_path)
        metadata = payload.get("metadata", {})
        run_summary = payload.get("run_summary", {})
        agent_traits = metadata.get("agent_traits", {})
        catalog.append(
            {
                "path": replay_path,
                "label": (
                    f"G{int(metadata.get('generation', 0)):03d}"
                    f" · R{int(metadata.get('run_index', 0)):03d}"
                    f" · {metadata.get('run_id', replay_path.stem)}"
                ),
                "run_id": str(metadata.get("run_id", replay_path.stem)),
                "generation": int(metadata.get("generation", 0)),
                "run_index": int(metadata.get("run_index", 0)),
                "lineage_id": str(metadata.get("lineage_id", "unknown")),
                "mode": str(metadata.get("mode", "unknown")),
                "experiment_id": str(metadata.get("experiment_id", "unknown")),
                "run_group": str(metadata.get("run_group", "unknown")),
                "traits": sorted({str(value) for value in agent_traits.values()}),
                "agent_traits": agent_traits,
                "metadata": metadata,
                "run_summary": run_summary,
            }
        )
    return catalog
