import json

from project_darwin.dashboard.terminal_view import TerminalRenderer
from project_darwin.experiments.configs.default import default_config
from project_darwin.experiments.run_manager import build_random_simulation
from project_darwin.simulation.scheduler import Scheduler


def main() -> None:
    config = default_config()
    world, agents = build_random_simulation(config)
    scheduler = Scheduler(config)

    with TerminalRenderer(
        delay_seconds=config.render_interval_seconds,
        recent_event_limit=config.recent_event_limit,
    ) as renderer:
        result = scheduler.run(world, agents, replay_name="terminal_run.json", on_turn=renderer.update)

    summary = {
        "turn": result.metrics.turn,
        "alive_agents": result.metrics.alive_agents,
        "total_messages": result.metrics.total_messages,
        "total_message_cost": result.metrics.total_message_cost,
        "total_forage_events": result.metrics.total_forage_events,
        "vocabulary_size": result.communication.vocabulary_size,
        "replay_path": str(result.replay_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
