import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from project_darwin.agents.base_agent import BaseAgent, ScriptedSurvivor
from project_darwin.agents.heuristic_agent import HeuristicSurvivor
from project_darwin.agents.llm_adapter import LLMAdapter
from project_darwin.agents.llm_agent import LLMSurvivor
from project_darwin.agents.random_agent import RandomSurvivor
from project_darwin.agents.traits import TraitProfile
from project_darwin.analytics.metrics_engine import MetricsEngine
from project_darwin.environment.resource_rules import build_initial_resources
from project_darwin.experiments.configs.default import default_config
from project_darwin.simulation.run_context import SimulationConfig
from project_darwin.simulation.scheduler import Scheduler
from project_darwin.simulation.state import AgentState, Position, WorldState


ABLATION_MODES = ("no_memory", "heuristic_only", "llm_with_memory")
BASELINE_BENCHMARK_GROUPS = (
    "scripted",
    "heuristic",
    "llm",
    "llm_without_memory",
    "llm_with_memory",
)
EXTENDED_BENCHMARK_GROUPS = BASELINE_BENCHMARK_GROUPS + (
    "llm_without_planning",
    "llm_without_social_reasoning",
)


def build_initial_world(
    config: SimulationConfig | None = None,
    spawn_points: dict[str, Position] | None = None,
) -> WorldState:
    config = config or default_config()
    spawn_points = spawn_points or {
        "agent_a": Position(0, 0),
        "agent_b": Position(2, 2),
        "agent_c": Position(4, 4),
    }

    world = WorldState(
        turn=0,
        width=config.width,
        height=config.height,
        agents={
            "agent_a": AgentState(
                agent_id="agent_a",
                family_id="alpha",
                position=spawn_points["agent_a"],
                energy=config.initial_energy,
            ),
            "agent_b": AgentState(
                agent_id="agent_b",
                family_id="beta",
                position=spawn_points["agent_b"],
                energy=config.initial_energy,
            ),
            "agent_c": AgentState(
                agent_id="agent_c",
                family_id="gamma",
                position=spawn_points["agent_c"],
                energy=config.initial_energy,
            ),
        },
        resources=build_initial_resources(config),
    )
    return world


def build_scripted_agents() -> dict[str, BaseAgent]:
    return {
        "agent_a": ScriptedSurvivor(agent_id="agent_a", family_id="alpha", trait=TraitProfile.COOPERATIVE),
        "agent_b": ScriptedSurvivor(agent_id="agent_b", family_id="beta", trait=TraitProfile.GREEDY),
        "agent_c": ScriptedSurvivor(agent_id="agent_c", family_id="gamma", trait=TraitProfile.SILENT),
    }


def build_scripted_simulation(config: SimulationConfig | None = None) -> tuple[WorldState, dict[str, BaseAgent]]:
    config = config or default_config()
    config.mode = "scripted"
    return build_initial_world(config), build_scripted_agents()


def build_random_simulation(config: SimulationConfig | None = None) -> tuple[WorldState, dict[str, BaseAgent]]:
    config = config or default_config()
    config.mode = "random"
    clustered_spawns = {
        "agent_a": Position(0, 0),
        "agent_b": Position(1, 0),
        "agent_c": Position(0, 1),
    }
    world = build_initial_world(config, spawn_points=clustered_spawns)
    agents: dict[str, BaseAgent] = {
        "agent_a": RandomSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=config.random_seed,
            message_probability=1.0,
        ),
        "agent_b": RandomSurvivor(
            agent_id="agent_b",
            family_id="beta",
            trait=TraitProfile.GREEDY,
            seed=config.random_seed + 1,
            message_probability=1.0,
        ),
        "agent_c": RandomSurvivor(
            agent_id="agent_c",
            family_id="gamma",
            trait=TraitProfile.SILENT,
            seed=config.random_seed + 2,
            message_probability=0.0,
        ),
    }
    return world, agents


def build_heuristic_agents(config: SimulationConfig) -> dict[str, BaseAgent]:
    return {
        "agent_a": HeuristicSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=config.random_seed,
            social_reasoning_enabled=config.social_reasoning_enabled,
            planning_enabled=config.planning_enabled,
        ),
        "agent_b": HeuristicSurvivor(
            agent_id="agent_b",
            family_id="beta",
            trait=TraitProfile.GREEDY,
            seed=config.random_seed + 1,
            social_reasoning_enabled=config.social_reasoning_enabled,
            planning_enabled=config.planning_enabled,
        ),
        "agent_c": HeuristicSurvivor(
            agent_id="agent_c",
            family_id="gamma",
            trait=TraitProfile.SILENT,
            seed=config.random_seed + 2,
            social_reasoning_enabled=config.social_reasoning_enabled,
            planning_enabled=config.planning_enabled,
        ),
    }


def build_heuristic_simulation(config: SimulationConfig | None = None) -> tuple[WorldState, dict[str, BaseAgent]]:
    config = config or default_config()
    config.mode = "heuristic"
    clustered_spawns = {
        "agent_a": Position(0, 0),
        "agent_b": Position(1, 0),
        "agent_c": Position(0, 1),
    }
    world = build_initial_world(config, spawn_points=clustered_spawns)
    return world, build_heuristic_agents(config)


def build_llm_agents(
    config: SimulationConfig,
    llm_adapter_factory: Callable[[str], Any] | None = None,
) -> dict[str, BaseAgent]:
    adapter_factory = llm_adapter_factory or (lambda _agent_id: LLMAdapter())
    return {
        "agent_a": LLMSurvivor(
            agent_id="agent_a",
            family_id="alpha",
            trait=TraitProfile.COOPERATIVE,
            seed=config.random_seed,
            llm_adapter=adapter_factory("agent_a"),
            planning_enabled=config.planning_enabled,
            social_reasoning_enabled=config.social_reasoning_enabled,
        ),
        "agent_b": LLMSurvivor(
            agent_id="agent_b",
            family_id="beta",
            trait=TraitProfile.GREEDY,
            seed=config.random_seed + 1,
            llm_adapter=adapter_factory("agent_b"),
            planning_enabled=config.planning_enabled,
            social_reasoning_enabled=config.social_reasoning_enabled,
        ),
        "agent_c": LLMSurvivor(
            agent_id="agent_c",
            family_id="gamma",
            trait=TraitProfile.SILENT,
            seed=config.random_seed + 2,
            llm_adapter=adapter_factory("agent_c"),
            planning_enabled=config.planning_enabled,
            social_reasoning_enabled=config.social_reasoning_enabled,
        ),
    }


def build_llm_simulation(
    config: SimulationConfig | None = None,
    llm_adapter_factory: Callable[[str], Any] | None = None,
) -> tuple[WorldState, dict[str, BaseAgent]]:
    config = config or default_config()
    config.mode = "llm"
    clustered_spawns = {
        "agent_a": Position(0, 0),
        "agent_b": Position(1, 0),
        "agent_c": Position(0, 1),
    }
    world = build_initial_world(config, spawn_points=clustered_spawns)
    return world, build_llm_agents(config, llm_adapter_factory=llm_adapter_factory)


def build_simulation(
    config: SimulationConfig,
    llm_adapter_factory: Callable[[str], Any] | None = None,
) -> tuple[WorldState, dict[str, BaseAgent]]:
    if config.mode == "random":
        return build_random_simulation(config)
    if config.mode == "heuristic":
        return build_heuristic_simulation(config)
    if config.mode == "llm":
        return build_llm_simulation(config, llm_adapter_factory=llm_adapter_factory)
    return build_scripted_simulation(config)


def apply_ablation_mode(config: SimulationConfig, ablation_mode: str) -> SimulationConfig:
    updated = replace(config)
    if ablation_mode == "llm_with_memory":
        updated.mode = "llm"
        updated.memory_enabled = True
        updated.trust_enabled = True
    elif ablation_mode == "heuristic_only":
        updated.mode = "heuristic"
        updated.memory_enabled = False
        updated.trust_enabled = False
    elif ablation_mode == "no_memory":
        updated.mode = "heuristic"
        updated.memory_enabled = False
        updated.trust_enabled = True
    else:
        raise ValueError(f"Unsupported ablation mode: {ablation_mode}")
    return updated


def _derive_seed(base_seed: int, generation: int, run_index: int) -> int:
    return base_seed + generation * 1000 + run_index


def _derive_benchmark_seed(base_seed: int, group_index: int, run_index: int) -> int:
    return base_seed + group_index * 10000 + run_index


def _experiment_root(config: SimulationConfig) -> Path:
    return config.artifact_dir / config.experiment_id / config.run_group


def configure_baseline_group(config: SimulationConfig, benchmark_group: str) -> SimulationConfig:
    updated = replace(config)
    updated.run_group = f"{config.run_group}_{benchmark_group}"
    updated.benchmark_group = benchmark_group

    if benchmark_group == "scripted":
        updated.mode = "scripted"
        updated.memory_enabled = False
        updated.trust_enabled = False
        return updated
    if benchmark_group == "heuristic":
        updated.mode = "heuristic"
        updated.memory_enabled = False
        updated.trust_enabled = True
        return updated
    if benchmark_group == "llm":
        updated.mode = "llm"
        return updated
    if benchmark_group == "llm_without_memory":
        updated.mode = "llm"
        updated.memory_enabled = False
        updated.trust_enabled = True
        return updated
    if benchmark_group == "llm_with_memory":
        updated.mode = "llm"
        updated.memory_enabled = True
        updated.trust_enabled = True
        return updated
    if benchmark_group == "llm_without_planning":
        updated.mode = "llm"
        updated.planning_enabled = False
        return updated
    if benchmark_group == "llm_without_social_reasoning":
        updated.mode = "llm"
        updated.social_reasoning_enabled = False
        updated.trust_enabled = False
        return updated
    raise ValueError(f"Unsupported benchmark group: {benchmark_group}")


def run_single_simulation(
    config: SimulationConfig,
    replay_name: str = "latest_run.json",
    llm_adapter_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    world, agents = build_simulation(config, llm_adapter_factory=llm_adapter_factory)
    scheduler = Scheduler(config)
    result = scheduler.run(world, agents, replay_name=replay_name)
    return {
        "mode": config.mode,
        "run_summary": result.run_summary.to_dict(),
        "replay_path": str(result.replay_path),
        "metadata": result.metadata,
    }


def run_generational_experiment(
    config: SimulationConfig,
    *,
    generations: int,
    runs_per_generation: int,
    ablation_mode: str,
    llm_adapter_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    metrics_engine = MetricsEngine()
    all_run_summaries: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    experiment_root = _experiment_root(config)
    experiment_root.mkdir(parents=True, exist_ok=True)

    for generation in range(generations):
        for run_index in range(runs_per_generation):
            run_config = replace(
                config,
                generation=generation,
                run_index=run_index,
                random_seed=_derive_seed(config.random_seed, generation, run_index),
            )
            run_config = apply_ablation_mode(run_config, ablation_mode)
            world, agents = build_simulation(run_config, llm_adapter_factory=llm_adapter_factory)
            scheduler = Scheduler(run_config)
            result = scheduler.run(world, agents, replay_name="replay.json", archive_run=True)
            run_summary = result.run_summary.to_dict()
            all_run_summaries.append(run_summary)
            run_records.append(
                {
                    "generation": generation,
                    "run_index": run_index,
                    "run_id": run_summary["run_id"],
                    "mode": run_summary["mode"],
                    "replay_path": str(result.replay_path),
                    "metadata": result.metadata,
                    "run_summary": run_summary,
                }
            )

    generation_summaries = metrics_engine.summarize_generations(all_run_summaries)
    manifest = {
        "experiment": {
            "experiment_id": config.experiment_id,
            "run_group": config.run_group,
            "lineage_id": config.lineage_id,
            "ablation_mode": ablation_mode,
            "base_seed": config.random_seed,
            "max_turns": config.max_turns,
            "generations": generations,
            "runs_per_generation": runs_per_generation,
        },
        "run_summaries": all_run_summaries,
        "generation_summaries": generation_summaries,
        "runs": run_records,
    }
    manifest_path = experiment_root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "experiment_root": str(experiment_root),
        "generation_summaries": generation_summaries,
        "run_count": len(run_records),
    }


def run_baseline_benchmark(
    config: SimulationConfig,
    *,
    runs_per_group: int,
    benchmark_groups: tuple[str, ...] = BASELINE_BENCHMARK_GROUPS,
    llm_adapter_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    metrics_engine = MetricsEngine()
    suite_root = _experiment_root(config)
    suite_root.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []

    for group_index, benchmark_group in enumerate(benchmark_groups):
        group_config = configure_baseline_group(config, benchmark_group)
        for run_index in range(runs_per_group):
            run_config = replace(
                group_config,
                generation=0,
                run_index=run_index,
                random_seed=_derive_benchmark_seed(config.random_seed, group_index, run_index),
            )
            world, agents = build_simulation(run_config, llm_adapter_factory=llm_adapter_factory)
            scheduler = Scheduler(run_config)
            result = scheduler.run(world, agents, replay_name="replay.json", archive_run=True)
            run_summary = result.run_summary.to_dict()
            run_summary["benchmark_group"] = benchmark_group
            run_summaries.append(run_summary)
            runs.append(
                {
                    "benchmark_group": benchmark_group,
                    "run_index": run_index,
                    "run_id": run_summary["run_id"],
                    "mode": run_summary["mode"],
                    "memory_enabled": run_summary["memory_enabled"],
                    "trust_enabled": run_summary["trust_enabled"],
                    "replay_path": str(result.replay_path),
                    "metadata": result.metadata,
                    "run_summary": run_summary,
                }
            )

    benchmark_summaries = metrics_engine.summarize_benchmark_groups(run_summaries)
    manifest = {
        "benchmark": {
            "experiment_id": config.experiment_id,
            "run_group": config.run_group,
            "lineage_id": config.lineage_id,
            "base_seed": config.random_seed,
            "max_turns": config.max_turns,
            "runs_per_group": runs_per_group,
            "benchmark_groups": list(benchmark_groups),
        },
        "group_summaries": benchmark_summaries,
        "run_summaries": run_summaries,
        "runs": runs,
    }
    manifest_path = suite_root / "baseline_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "experiment_root": str(suite_root),
        "group_summaries": benchmark_summaries,
        "run_count": len(runs),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project Darwin experiment runner")
    parser.add_argument("--mode", choices=["scripted", "random", "heuristic", "llm"], default="heuristic")
    parser.add_argument("--experiment-id", default="default_experiment")
    parser.add_argument("--run-group", default="default_run_group")
    parser.add_argument("--lineage-id", default="mixed_population")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--runs-per-generation", type=int, default=1)
    parser.add_argument("--benchmark-baselines", action="store_true")
    parser.add_argument("--benchmark-extended", action="store_true")
    parser.add_argument("--runs-per-group", type=int, default=30)
    parser.add_argument("--disable-planning", action="store_true")
    parser.add_argument("--disable-social-reasoning", action="store_true")
    parser.add_argument("--ablation-mode", choices=ABLATION_MODES, default="no_memory")
    parser.add_argument("--archive-experiment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = default_config()
    config.mode = args.mode
    config.experiment_id = args.experiment_id
    config.run_group = args.run_group
    config.lineage_id = args.lineage_id
    config.artifact_dir = Path(args.artifact_dir)
    config.random_seed = args.seed
    config.max_turns = args.max_turns
    config.planning_enabled = not args.disable_planning
    config.social_reasoning_enabled = not args.disable_social_reasoning

    if args.benchmark_baselines:
        summary = run_baseline_benchmark(
            config,
            runs_per_group=args.runs_per_group,
            benchmark_groups=EXTENDED_BENCHMARK_GROUPS if args.benchmark_extended else BASELINE_BENCHMARK_GROUPS,
        )
    elif args.archive_experiment or args.generations > 1 or args.runs_per_generation > 1:
        summary = run_generational_experiment(
            config,
            generations=args.generations,
            runs_per_generation=args.runs_per_generation,
            ablation_mode=args.ablation_mode,
        )
    else:
        summary = run_single_simulation(config)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
