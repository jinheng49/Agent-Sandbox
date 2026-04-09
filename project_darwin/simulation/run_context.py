from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SimulationConfig:
    experiment_id: str = "default_experiment"
    run_group: str = "default_run_group"
    benchmark_group: str = ""
    run_index: int = 0
    width: int = 10
    height: int = 10
    max_turns: int = 25
    stop_when_one_agent_remains: bool = True
    decision_workers: int = 4
    observation_radius: int = 1
    random_seed: int = 7
    generation: int = 0
    lineage_id: str = "mixed_population"
    mode: str = "scripted"
    initial_energy: int = 12
    move_cost: int = 1
    message_cost_per_char: int = 1
    forage_gain: int = 4
    gold_gain: int = 8
    forage_nodes: int = 12
    gold_nodes: int = 2
    competitive_resource_layout: bool = False
    render_interval_seconds: float = 0.12
    recent_event_limit: int = 10
    memory_enabled: bool = True
    memory_collection_name: str = "evolution_memory"
    memory_vector_size: int = 48
    memory_limit: int = 3
    memory_score_threshold: float = 0.15
    reflection_window: int = 20
    trust_enabled: bool = True
    trust_window: int = 4
    trust_reward: float = 0.6
    trust_penalty: float = 0.9
    deception_enabled: bool = True
    social_reasoning_enabled: bool = True
    planning_enabled: bool = True
    artifact_dir: Path = Path("artifacts")
