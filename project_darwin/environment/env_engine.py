from project_darwin.agents.action_space import ActionType, AgentAction, Direction
from project_darwin.simulation.event_bus import EventBus, EventType
from project_darwin.simulation.run_context import SimulationConfig
from project_darwin.simulation.state import AgentState, Position, ResourceType, WorldState


class EnvironmentEngine:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def step(self, world: WorldState, actions: dict[str, AgentAction], event_bus: EventBus) -> WorldState:
        next_turn = world.turn + 1

        for agent_id in sorted(world.agents):
            agent = world.agents[agent_id]
            if not agent.alive:
                continue

            action = actions.get(agent_id, AgentAction(action_type=ActionType.REST))
            self._apply_action(world, agent, action, event_bus, next_turn)

            if agent.energy <= 0 and agent.alive:
                agent.alive = False
                event_bus.record(
                    turn=next_turn,
                    event_type=EventType.DEATH,
                    agent_id=agent_id,
                    family_id=agent.family_id,
                    payload={
                        "reason": "energy_depletion",
                        "energy": agent.energy,
                        "position": {"x": agent.position.x, "y": agent.position.y},
                    },
                )

        world.turn = next_turn
        return world

    def _apply_action(
        self,
        world: WorldState,
        agent: AgentState,
        action: AgentAction,
        event_bus: EventBus,
        turn: int,
    ) -> None:
        plan_payload = self._plan_payload(action)
        if action.action_type is ActionType.MOVE and action.direction is not None:
            self._move_agent(world, agent, action.direction)
            agent.energy -= self.config.move_cost
            event_bus.record(
                turn=turn,
                event_type=EventType.MOVE,
                agent_id=agent.agent_id,
                family_id=agent.family_id,
                payload={
                    "direction": action.direction.value,
                    "position": {"x": agent.position.x, "y": agent.position.y},
                    "cost": self.config.move_cost,
                    **plan_payload,
                },
            )
            return

        if action.action_type is ActionType.MESSAGE and action.content:
            message_cost = len(action.content) * self.config.message_cost_per_char
            agent.energy -= message_cost
            target_position = None
            if action.message_target is not None:
                target_position = {"x": action.message_target[0], "y": action.message_target[1]}
            event_bus.record(
                turn=turn,
                event_type=EventType.MESSAGE,
                agent_id=agent.agent_id,
                family_id=agent.family_id,
                payload={
                    "message": {
                        "sender_id": agent.agent_id,
                        "sender_family_id": agent.family_id,
                        "content": action.content,
                        "content_length": len(action.content),
                        "channel": "broadcast",
                        "intent": action.message_intent.value,
                        "target_position": target_position,
                        "resource_hint": action.resource_hint,
                    },
                    "cost": message_cost,
                    **plan_payload,
                },
            )
            return

        if action.action_type is ActionType.FORAGE:
            resource = world.resources.pop(agent.position, None)
            if resource is None:
                event_bus.record(
                    turn=turn,
                    event_type=EventType.FORAGE_MISS,
                    agent_id=agent.agent_id,
                    family_id=agent.family_id,
                    payload={"position": {"x": agent.position.x, "y": agent.position.y}},
                )
                return

            gain = self.config.forage_gain if resource.kind is ResourceType.FOOD else self.config.gold_gain
            agent.inventory[resource.kind.value] = agent.inventory.get(resource.kind.value, 0) + 1
            cooperation_payload = None
            if resource.kind is ResourceType.GOLD and action.share_with_nearby:
                cooperation_payload = self._share_gold_with_nearby(world, agent, gain, turn, event_bus)
            else:
                agent.energy += gain

            event_bus.record(
                turn=turn,
                event_type=EventType.FORAGE,
                agent_id=agent.agent_id,
                family_id=agent.family_id,
                payload={
                    "resource": resource.kind.value,
                    "gain": cooperation_payload["collector_gain"] if cooperation_payload is not None else gain,
                    "position": {"x": agent.position.x, "y": agent.position.y},
                    "shared": cooperation_payload is not None,
                    "shared_with": cooperation_payload["participant_ids"] if cooperation_payload is not None else [],
                    **plan_payload,
                },
            )
            return

        event_bus.record(
            turn=turn,
            event_type=EventType.REST,
            agent_id=agent.agent_id,
            family_id=agent.family_id,
            payload={"position": {"x": agent.position.x, "y": agent.position.y}, **plan_payload},
        )

    def _plan_payload(self, action: AgentAction) -> dict[str, object]:
        target_position = None
        if action.planned_target_position is not None:
            target_position = {
                "x": action.planned_target_position[0],
                "y": action.planned_target_position[1],
            }
        return {
            "decision_source": action.decision_source,
            "decision_note": action.decision_note,
            "current_goal": action.current_goal,
            "planned_target_position": target_position,
        }

    def _move_agent(self, world: WorldState, agent: AgentState, direction: Direction) -> None:
        delta_x = 0
        delta_y = 0
        if direction is Direction.UP:
            delta_y = -1
        elif direction is Direction.DOWN:
            delta_y = 1
        elif direction is Direction.LEFT:
            delta_x = -1
        elif direction is Direction.RIGHT:
            delta_x = 1

        next_x = (agent.position.x + delta_x) % world.width
        next_y = (agent.position.y + delta_y) % world.height
        agent.position = Position(next_x, next_y)

    def _share_gold_with_nearby(
        self,
        world: WorldState,
        collector: AgentState,
        gain: int,
        turn: int,
        event_bus: EventBus,
    ) -> dict[str, object]:
        nearby_agents = self._nearby_alive_agents(world, collector)
        participants = [collector, *nearby_agents]
        gain_per_agent = max(1, gain // len(participants))

        # Gold sharing is the first concrete cooperation mechanic; later stages can attach trust and betrayal to it.
        for participant in participants:
            participant.energy += gain_per_agent

        participant_ids = [participant.agent_id for participant in participants if participant.agent_id != collector.agent_id]
        event_bus.record(
            turn=turn,
            event_type=EventType.COOPERATION,
            agent_id=collector.agent_id,
            family_id=collector.family_id,
            payload={
                "kind": "gold_share",
                "position": {"x": collector.position.x, "y": collector.position.y},
                "participant_ids": participant_ids,
                "gain_per_agent": gain_per_agent,
            },
        )
        return {
            "collector_gain": gain_per_agent,
            "participant_ids": participant_ids,
        }

    def _nearby_alive_agents(self, world: WorldState, agent: AgentState, radius: int = 1) -> list[AgentState]:
        return [
            other_agent
            for other_id, other_agent in world.agents.items()
            if other_id != agent.agent_id
            and other_agent.alive
            and abs(other_agent.position.x - agent.position.x) <= radius
            and abs(other_agent.position.y - agent.position.y) <= radius
        ]
