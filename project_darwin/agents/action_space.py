import json
from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    REST = "rest"
    MOVE = "move"
    MESSAGE = "message"
    FORAGE = "forage"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class MessageIntent(str, Enum):
    CONTACT = "contact"
    SHARE_FOOD = "share_food"
    SHARE_GOLD = "share_gold"
    CLAIM_GOLD = "claim_gold"
    FALSE_GOLD = "false_gold"
    STAY_SILENT = "stay_silent"


@dataclass(frozen=True, slots=True)
class AgentAction:
    action_type: ActionType
    direction: Direction | None = None
    content: str = ""
    message_intent: MessageIntent = MessageIntent.CONTACT
    message_target: tuple[int, int] | None = None
    resource_hint: str | None = None
    share_with_nearby: bool = False
    decision_source: str = "native"
    decision_note: str = ""
    current_goal: str = ""
    planned_target_position: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ShortTermPlan:
    current_goal: str = ""
    planned_target_position: tuple[int, int] | None = None
    created_turn: int = -1

    def is_empty(self) -> bool:
        return not self.current_goal and self.planned_target_position is None


def action_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action_type"],
        "properties": {
            "action_type": {"type": "string", "enum": [action_type.value for action_type in ActionType]},
            "direction": {"type": ["string", "null"], "enum": [direction.value for direction in Direction] + [None]},
            "content": {"type": "string"},
            "message_intent": {"type": "string", "enum": [intent.value for intent in MessageIntent]},
            "message_target": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["x", "y"],
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
            },
            "resource_hint": {"type": ["string", "null"]},
            "share_with_nearby": {"type": "boolean"},
            "current_goal": {"type": "string"},
            "planned_target_position": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["x", "y"],
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
            },
        },
    }


def action_schema_text() -> str:
    return json.dumps(action_json_schema(), indent=2, ensure_ascii=True)


def parse_action_text(raw_text: str) -> AgentAction:
    payload = json.loads(_extract_json_object(raw_text))
    return agent_action_from_payload(payload)


def agent_action_from_payload(payload: dict[str, object]) -> AgentAction:
    if "action_type" not in payload:
        raise ValueError("action_type is required")

    action_type = ActionType(str(payload["action_type"]))
    direction_payload = payload.get("direction")
    direction = Direction(str(direction_payload)) if direction_payload not in (None, "") else None
    content = str(payload.get("content", ""))
    message_intent = MessageIntent(str(payload.get("message_intent", MessageIntent.CONTACT.value)))
    message_target = _parse_target(payload.get("message_target"))
    resource_hint = payload.get("resource_hint")
    share_with_nearby = bool(payload.get("share_with_nearby", False))
    current_goal = str(payload.get("current_goal", "")).strip()
    planned_target_position = _parse_target(payload.get("planned_target_position"))

    if action_type is ActionType.MOVE and direction is None:
        raise ValueError("direction is required for move actions")
    if action_type is ActionType.MESSAGE and not content:
        raise ValueError("content is required for message actions")

    return AgentAction(
        action_type=action_type,
        direction=direction,
        content=content,
        message_intent=message_intent,
        message_target=message_target,
        resource_hint=str(resource_hint) if resource_hint is not None else None,
        share_with_nearby=share_with_nearby,
        current_goal=current_goal,
        planned_target_position=planned_target_position,
    )


def _parse_target(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict) and {"x", "y"}.issubset(value):
        return (int(value["x"]), int(value["y"]))
    raise ValueError("message_target must be an object with x and y")


def _extract_json_object(raw_text: str) -> str:
    sanitized = "".join(character for character in raw_text if character in {"\n", "\t"} or ord(character) >= 32)
    start = sanitized.find("{")
    end = sanitized.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return sanitized[start : end + 1]
