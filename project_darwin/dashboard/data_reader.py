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


def discover_experiment_manifest_paths(root: Path | None = None) -> list[Path]:
    artifact_root = root or Path("artifacts")
    if not artifact_root.exists():
        return []
    return sorted(candidate for candidate in artifact_root.rglob("experiment_manifest.json") if candidate.is_file())


def build_experiment_catalog(root: Path | None = None) -> list[dict[str, Any]]:
    artifact_root = root or Path("artifacts")
    catalog: list[dict[str, Any]] = []
    for manifest_path in discover_experiment_manifest_paths(artifact_root):
        payload = load_replay(manifest_path)
        experiment = payload.get("experiment", {})
        generation_summaries = sorted(
            payload.get("generation_summaries", []),
            key=lambda row: int(row.get("generation", 0)),
        )
        latest_summary = generation_summaries[-1] if generation_summaries else {}
        catalog.append(
            {
                "path": manifest_path,
                "label": (
                    f"{experiment.get('experiment_id', manifest_path.parent.name)}"
                    f" · {experiment.get('run_group', 'unknown')}"
                    f" · {experiment.get('ablation_mode', 'unknown')}"
                ),
                "experiment_id": str(experiment.get("experiment_id", manifest_path.parent.name)),
                "run_group": str(experiment.get("run_group", "unknown")),
                "lineage_id": str(experiment.get("lineage_id", "unknown")),
                "ablation_mode": str(experiment.get("ablation_mode", "unknown")),
                "generations": int(experiment.get("generations", len(generation_summaries))),
                "runs_per_generation": int(experiment.get("runs_per_generation", 0)),
                "manifest": payload,
                "generation_summaries": generation_summaries,
                "latest_summary": latest_summary,
            }
        )
    return catalog


def build_generation_metric_rows(manifest: dict[str, Any], metric_name: str) -> list[dict[str, float | int]]:
    rows = sorted(manifest.get("generation_summaries", []), key=lambda row: int(row.get("generation", 0)))
    return [
        {
            "generation": int(row.get("generation", 0)),
            metric_name: float(row.get(metric_name, 0.0)),
        }
        for row in rows
    ]
