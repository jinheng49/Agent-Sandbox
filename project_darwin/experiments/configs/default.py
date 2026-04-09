from project_darwin.simulation.run_context import SimulationConfig


def default_config() -> SimulationConfig:
    return SimulationConfig(
        width=6,
        height=6,
        observation_radius=2,
        initial_energy=10,
        forage_nodes=8,
        gold_nodes=4,
        competitive_resource_layout=False,
    )
