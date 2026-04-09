from project_darwin.simulation.run_context import SimulationConfig
from project_darwin.simulation.state import Position, ResourceNode, ResourceType


def build_initial_resources(config: SimulationConfig) -> dict[Position, ResourceNode]:
    if config.competitive_resource_layout:
        return _build_competitive_resources(config)

    resources: dict[Position, ResourceNode] = {}

    for index in range(config.forage_nodes):
        position = Position(index % config.width, (index * 3) % config.height)
        resources[position] = ResourceNode(kind=ResourceType.FOOD, amount=1)

    for index in range(config.gold_nodes):
        position = Position((config.width - 1 - index) % config.width, (index * 4 + 1) % config.height)
        resources[position] = ResourceNode(kind=ResourceType.GOLD, amount=1)

    return resources


def _build_competitive_resources(config: SimulationConfig) -> dict[Position, ResourceNode]:
    resources: dict[Position, ResourceNode] = {}
    center_x = config.width // 2
    center_y = config.height // 2

    gold_candidates = [
        Position(center_x, center_y),
        Position(max(center_x - 1, 0), center_y),
        Position(min(center_x + 1, config.width - 1), center_y),
        Position(center_x, max(center_y - 1, 0)),
        Position(center_x, min(center_y + 1, config.height - 1)),
    ]
    food_candidates = [
        Position(max(center_x - 2, 0), center_y),
        Position(min(center_x + 2, config.width - 1), center_y),
        Position(center_x, max(center_y - 2, 0)),
        Position(center_x, min(center_y + 2, config.height - 1)),
        Position(max(center_x - 1, 0), max(center_y - 1, 0)),
        Position(min(center_x + 1, config.width - 1), max(center_y - 1, 0)),
        Position(max(center_x - 1, 0), min(center_y + 1, config.height - 1)),
        Position(min(center_x + 1, config.width - 1), min(center_y + 1, config.height - 1)),
        Position(center_x, center_y),
    ]

    for position in gold_candidates[: config.gold_nodes]:
        resources[position] = ResourceNode(kind=ResourceType.GOLD, amount=1)

    for position in food_candidates:
        if len([node for node in resources.values() if node.kind is ResourceType.FOOD]) >= config.forage_nodes:
            break
        if position in resources:
            continue
        resources[position] = ResourceNode(kind=ResourceType.FOOD, amount=1)

    if len([node for node in resources.values() if node.kind is ResourceType.FOOD]) < config.forage_nodes:
        for index in range(config.forage_nodes):
            position = Position(index % config.width, (index * 3) % config.height)
            if position in resources:
                continue
            resources[position] = ResourceNode(kind=ResourceType.FOOD, amount=1)
            if len([node for node in resources.values() if node.kind is ResourceType.FOOD]) >= config.forage_nodes:
                break

    return resources
