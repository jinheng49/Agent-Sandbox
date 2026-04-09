from collections import Counter
from time import sleep

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from project_darwin.agents.action_space import ActionType, AgentAction
from project_darwin.simulation.event_bus import EventBus
from project_darwin.simulation.state import Position, ResourceType, WorldState


class TerminalRenderer:
    def __init__(self, delay_seconds: float = 0.12, recent_event_limit: int = 10) -> None:
        self.delay_seconds = delay_seconds
        self.recent_event_limit = recent_event_limit
        self.live: Live | None = None

    def __enter__(self) -> "TerminalRenderer":
        self.live = Live(auto_refresh=False, transient=False)
        self.live.__enter__()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.live is not None:
            self.live.__exit__(exc_type, exc_value, traceback)

    def update(self, world: WorldState, event_bus: EventBus, actions: dict[str, AgentAction]) -> None:
        if self.live is None:
            return

        self.live.update(self._build_layout(world, event_bus, actions), refresh=True)
        if self.delay_seconds > 0:
            sleep(self.delay_seconds)

    def _build_layout(self, world: WorldState, event_bus: EventBus, actions: dict[str, AgentAction]) -> Group:
        return Group(
            Panel(self._build_summary_table(world, event_bus), title="Simulation Summary"),
            Panel(self._build_grid_table(world), title="World Grid"),
            Panel(self._build_agent_table(world, actions), title="Agents"),
            Panel(self._build_event_table(event_bus), title="Recent Events"),
        )

    def _build_summary_table(self, world: WorldState, event_bus: EventBus) -> Table:
        event_counts = Counter(event.event_type.value for event in event_bus.events)
        table = Table(show_header=False, box=None)
        table.add_row("Turn", str(world.turn))
        table.add_row("Alive", str(sum(1 for agent in world.agents.values() if agent.alive)))
        table.add_row("Resources", str(len(world.resources)))
        table.add_row("Messages", str(event_counts.get("message", 0)))
        table.add_row("Forage", str(event_counts.get("forage", 0)))
        table.add_row("Deaths", str(event_counts.get("death", 0)))
        return table

    def _build_grid_table(self, world: WorldState) -> Table:
        table = Table(show_header=False, box=None, pad_edge=False)
        agent_lookup = self._build_agent_lookup(world)

        for y_coord in range(world.height):
            row = []
            for x_coord in range(world.width):
                position = Position(x_coord, y_coord)
                cell_value = "."
                if position in world.resources:
                    resource = world.resources[position]
                    cell_value = "G" if resource.kind is ResourceType.GOLD else "F"
                if position in agent_lookup:
                    cell_value = agent_lookup[position]
                row.append(cell_value)
            table.add_row(" ".join(row))

        return table

    def _build_agent_lookup(self, world: WorldState) -> dict[Position, str]:
        agent_lookup: dict[Position, str] = {}
        for agent_id, agent in world.agents.items():
            if not agent.alive:
                continue
            marker = agent_id.split("_")[-1][0].upper()
            if agent.position in agent_lookup:
                marker = "*"
            agent_lookup[agent.position] = marker
        return agent_lookup

    def _build_agent_table(self, world: WorldState, actions: dict[str, AgentAction]) -> Table:
        table = Table(show_header=True)
        table.add_column("Agent")
        table.add_column("Energy")
        table.add_column("Position")
        table.add_column("Alive")
        table.add_column("Last Action")
        table.add_column("Inventory")

        for agent_id, agent in world.agents.items():
            inventory = ", ".join(f"{name}:{count}" for name, count in sorted(agent.inventory.items())) or "-"
            table.add_row(
                agent_id,
                str(agent.energy),
                f"({agent.position.x}, {agent.position.y})",
                str(agent.alive),
                self._format_action(actions.get(agent_id)),
                inventory,
            )

        return table

    def _build_event_table(self, event_bus: EventBus) -> Table:
        table = Table(show_header=True)
        table.add_column("Turn")
        table.add_column("Type")
        table.add_column("Payload")

        for event in event_bus.events[-self.recent_event_limit :]:
            payload = ", ".join(f"{key}={value}" for key, value in sorted(event.payload.items()))
            table.add_row(str(event.turn), event.event_type.value, payload)

        return table

    def _format_action(self, action: AgentAction | None) -> str:
        if action is None:
            return "-"
        if action.action_type is ActionType.MOVE and action.direction is not None:
            return f"move:{action.direction.value}"
        if action.action_type is ActionType.MESSAGE:
            return f"message:{action.content}"
        return action.action_type.value
