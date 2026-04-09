import sys
import time
from html import escape
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import pandas as pd

from project_darwin.agents.llm_adapter import LLMAdapter
from project_darwin.dashboard.data_reader import (
    build_experiment_catalog,
    build_generation_metric_rows,
    build_replay_catalog,
    load_replay,
)
from project_darwin.experiments.configs.default import default_config
from project_darwin.experiments.run_manager import (
    build_heuristic_simulation,
    build_llm_simulation,
    build_random_simulation,
    build_scripted_simulation,
)
from project_darwin.simulation.scheduler import Scheduler
from project_darwin.simulation.state import Position, ResourceType, WorldState, deserialize_world_state


AGENT_COLORS = {
    "agent_a": "#c84c2f",
    "agent_b": "#1f6b5a",
    "agent_c": "#1f4e79",
}

TRAIT_LABELS = {
    "cooperative": "协作型",
    "greedy": "贪婪型",
    "silent": "寡言型",
}

TRAIT_DESCRIPTIONS = {
    "cooperative": "更愿意通信与协作，倾向共享高价值线索。",
    "greedy": "优先追逐高价值资源，更激进地争夺收益。",
    "silent": "低通信、低暴露，偏向独立生存和稳健移动。",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --paper: #f4efe4;
            --ink: #162126;
            --muted: #5b665e;
            --panel: rgba(255, 252, 245, 0.88);
            --line: rgba(22, 33, 38, 0.12);
            --gold: #d5a021;
            --food: #6e9f52;
            --empty: #ece5d8;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(213,160,33,0.16), transparent 32%),
                radial-gradient(circle at top right, rgba(31,107,90,0.14), transparent 28%),
                linear-gradient(180deg, #f8f3e8 0%, #efe7d7 100%);
            color: var(--ink);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        .hero {
            padding: 1.1rem 1.3rem;
            border: 1px solid var(--line);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,252,245,0.94), rgba(243,233,214,0.9));
            box-shadow: 0 18px 50px rgba(73, 57, 25, 0.07);
            margin-bottom: 1rem;
        }

        .hero-kicker {
            font-size: 0.76rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8b6c25;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .hero-title {
            font-size: 2rem;
            line-height: 1.05;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .hero-copy {
            color: var(--muted);
            line-height: 1.55;
            max-width: 54rem;
        }

        .panel-shell {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: 0 12px 36px rgba(22, 33, 38, 0.06);
            margin-bottom: 1rem;
        }

        .section-heading {
            font-size: 0.98rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            color: #7b5d1d;
            margin-bottom: 0.25rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }

        .step-bar {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .step-card {
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
        }

        .step-label {
            color: var(--muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }

        .step-value {
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1;
        }

        .map-wrap {
            display: grid;
            grid-template-columns: repeat(10, minmax(34px, 1fr));
            gap: 0.35rem;
        }

        .map-cell {
            position: relative;
            aspect-ratio: 1 / 1;
            border-radius: 14px;
            border: 1px solid rgba(22, 33, 38, 0.08);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
        }

        .map-cell-empty { background: var(--empty); color: #9b927d; }
        .map-cell-food { background: rgba(110,159,82,0.18); color: #40612f; }
        .map-cell-gold { background: rgba(213,160,33,0.24); color: #8c6512; }
        .map-cell-agent { color: white; }
        .map-cell-overlap { background: linear-gradient(135deg, #b6452d, #194e7a); color: white; }
        .map-cell-moved {
            outline: 3px solid rgba(255, 142, 61, 0.82);
            transform: scale(1.04);
            box-shadow: 0 0 0 6px rgba(255, 142, 61, 0.16);
        }

        .cell-coord {
            position: absolute;
            left: 6px;
            top: 4px;
            font-size: 0.56rem;
            opacity: 0.42;
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 500;
        }

        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.9rem;
        }

        .legend-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.64);
            border: 1px solid var(--line);
            font-size: 0.8rem;
        }

        .legend-swatch {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
        }

        .action-stack {
            display: grid;
            gap: 0.75rem;
        }

        .action-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(247,241,230,0.92));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(22, 33, 38, 0.05);
        }

        .action-card-message {
            border-color: rgba(31, 78, 121, 0.18);
            background: linear-gradient(180deg, rgba(236,244,251,0.96), rgba(248,250,252,0.96));
            box-shadow: 0 14px 34px rgba(31, 78, 121, 0.08);
        }

        .action-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            margin-bottom: 0.65rem;
        }

        .action-card-agent {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            font-weight: 700;
        }

        .action-card-dot {
            width: 11px;
            height: 11px;
            border-radius: 999px;
            display: inline-block;
            box-shadow: 0 0 0 4px rgba(255,255,255,0.72);
        }

        .action-card-meta {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.45rem;
        }

        .action-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.26rem 0.58rem;
            border-radius: 999px;
            background: rgba(22, 33, 38, 0.07);
            color: var(--ink);
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .action-pill-message {
            background: rgba(31, 78, 121, 0.12);
            color: #163d60;
        }

        .action-summary {
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.45;
            margin-bottom: 0.55rem;
        }

        .action-agent-profile {
            margin-bottom: 0.6rem;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .action-message-box {
            border-left: 4px solid #1f4e79;
            background: rgba(31, 78, 121, 0.07);
            border-radius: 12px;
            padding: 0.72rem 0.8rem;
            margin-bottom: 0.55rem;
        }

        .action-message-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #44627d;
            margin-bottom: 0.28rem;
            font-weight: 700;
        }

        .action-message-content {
            font-size: 1.03rem;
            line-height: 1.5;
            color: #163d60;
            font-weight: 700;
            word-break: break-word;
        }

        .action-footer {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">Project Darwin</div>
            <div class="hero-title">单地图行动工作台</div>
            <div class="hero-copy">主界面只保留一个核心地图。无论是模拟模式还是复盘模式，当前这一步每个 agent 的行动都会在地图旁边直接给出。</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _describe_action(action: Any) -> str:
    if getattr(action, "action_type", None) is None:
        return "-"
    if action.action_type.value == "move":
        return f"移动到 {action.direction.value}"
    if action.action_type.value == "forage":
        return "采集资源并共享" if getattr(action, "share_with_nearby", False) else "采集资源"
    if action.action_type.value == "message":
        content = getattr(action, "content", "") or ""
        target = f"，目标 {action.message_target}" if getattr(action, "message_target", None) is not None else ""
        return f"发送消息“{content}”，意图 {action.message_intent.value}{target}"
    return "休息"


def _describe_event(event: dict[str, Any]) -> str:
    event_type = event.get("event_type", "unknown")
    payload = event.get("payload", {})
    if event_type == "message":
        message = payload.get("message", {})
        content = message.get("content", "")
        intent = message.get("intent", "contact")
        target = message.get("target_position")
        target_text = f"，目标 {target}" if target is not None else ""
        return f"发送消息“{content}”，意图 {intent}{target_text}"
    if event_type == "move":
        return f"向 {payload.get('direction', '-')} 移动到 {payload.get('position', {})}"
    if event_type == "forage":
        return f"采集 {payload.get('resource', '-')}，收益 {payload.get('gain', 0)}"
    if event_type == "forage_miss":
        return "尝试采集，但当前位置没有资源"
    if event_type == "cooperation":
        return f"触发合作分账，参与者 {payload.get('participant_ids', [])}"
    if event_type == "death":
        return f"死亡，原因 {payload.get('reason', '-')}"
    if event_type == "rest":
        return "休息"
    if event_type == "trust_update":
        return f"信任变化 {payload.get('delta', 0)}"
    return event_type


def _trait_profile_text(trait: str) -> tuple[str, str]:
    return TRAIT_LABELS.get(trait, trait or "未知"), TRAIT_DESCRIPTIONS.get(trait, "当前没有额外性格说明。")


def _build_live_action_rows(actions: dict[str, Any], agents: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for agent_id, action in sorted(actions.items()):
        agent = agents.get(agent_id)
        trait_value = getattr(getattr(agent, "trait", None), "value", "unknown")
        trait_label, persona = _trait_profile_text(trait_value)
        action_type = getattr(action, "action_type", None)
        is_message = bool(action_type and action_type.value == "message")
        message_content = getattr(action, "content", "") or ""
        target = getattr(action, "message_target", None)
        target_text = str(target) if target is not None else "-"
        intent = getattr(getattr(action, "message_intent", None), "value", "-")
        rows.append(
            {
                "agent": agent_id,
                "action": action_type.value if action_type else "-",
                "detail": _describe_action(action),
                "source": getattr(action, "decision_source", "native"),
                "note": getattr(action, "decision_note", "") or "-",
                "is_message": "1" if is_message else "0",
                "message": message_content,
                "intent": intent,
                "target": target_text,
                "goal": getattr(action, "current_goal", "") or "-",
                "planned_target": str(getattr(action, "planned_target_position", None) or "-"),
                "social": "-",
                "family": getattr(agent, "family_id", "-"),
                "trait": trait_label,
                "persona": persona,
            }
        )
    return rows


def _build_replay_action_rows(
    events: list[dict[str, Any]],
    turn: int,
    metadata: dict[str, Any],
    world: WorldState,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    agent_traits = metadata.get("agent_traits", {}) if isinstance(metadata, dict) else {}
    current_turn_events = [event for event in events if int(event.get("turn", 0)) == turn]
    for event in current_turn_events:
        agent_id = event.get("agent_id")
        if not agent_id or event.get("event_type") in {"turn_start", "turn_end"}:
            continue
        agent_state = world.agents.get(str(agent_id))
        trait_value = str(agent_traits.get(str(agent_id), "unknown"))
        trait_label, persona = _trait_profile_text(trait_value)
        payload = event.get("payload", {})
        message = payload.get("message", {}) if isinstance(payload, dict) else {}
        is_message = event.get("event_type") == "message"
        target = message.get("target_position") if is_message else None
        rows.append(
            {
                "agent": str(agent_id),
                "action": str(event.get("event_type", "-")),
                "detail": _describe_event(event),
                "source": str(payload.get("decision_source", "replay")),
                "note": str(payload.get("decision_note", "-")),
                "is_message": "1" if is_message else "0",
                "message": str(message.get("content", "")) if is_message else "",
                "intent": str(message.get("intent", "-")) if is_message else "-",
                "target": str(target) if target is not None else "-",
                "goal": str(payload.get("current_goal", "-")) or "-",
                "planned_target": str(payload.get("planned_target_position", "-")),
                "social": _social_badge_text(event),
                "family": "-" if agent_state is None else agent_state.family_id,
                "trait": trait_label,
                "persona": persona,
            }
        )
    return rows


def _social_badge_text(event: dict[str, Any]) -> str:
    payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
    if event.get("event_type") == "trust_update":
        delta = payload.get("delta", 0)
        score = payload.get("score", 0)
        return f"信任 {delta:+} / 当前 {score}"
    message = payload.get("message", {}) if isinstance(payload.get("message"), dict) else {}
    if message:
        return f"社会线索 {message.get('intent', '-') }"
    return "-"


def _render_action_panel(action_rows: list[dict[str, str]], title: str) -> None:
    st.markdown(
        f'<div class="panel-shell"><div class="section-heading">{title}</div><div class="section-copy">每一步只展示当前地图对应的 agent 行动，消息会单独高亮显示。</div>',
        unsafe_allow_html=True,
    )
    if not action_rows:
        st.info("当前这一步没有 agent 行动。")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    cards: list[str] = []
    for row in action_rows:
        agent = escape(row["agent"])
        action = escape(row["action"])
        detail = escape(row["detail"])
        source = escape(row["source"])
        note = escape(row.get("note", "-"))
        goal = escape(row.get("goal", "-") or "-")
        planned_target = escape(row.get("planned_target", "-") or "-")
        social = escape(row.get("social", "-") or "-")
        family = escape(row.get("family", "-") or "-")
        trait = escape(row.get("trait", "-") or "-")
        persona = escape(row.get("persona", "-") or "-")
        is_message = row.get("is_message") == "1"
        color = AGENT_COLORS.get(row["agent"], "#444")
        card_class = "action-card action-card-message" if is_message else "action-card"
        pill_class = "action-pill action-pill-message" if is_message else "action-pill"
        message_box = ""
        if is_message:
            message_content = escape(row.get("message", "") or "（空消息）")
            intent = escape(row.get("intent", "-"))
            target = escape(row.get("target", "-"))
            message_box = (
                '<div class="action-message-box">'
                '<div class="action-message-label">谁发了什么消息</div>'
                f'<div class="action-message-content">{message_content}</div>'
                '</div>'
            )
            footer = (
                '<div class="action-footer">'
                f'<span class="action-pill">意图 {intent}</span>'
                f'<span class="action-pill">目标 {target}</span>'
                f'<span class="action-pill">计划 {goal}</span>'
                f'<span class="action-pill">规划坐标 {planned_target}</span>'
                '</div>'
            )
        else:
            footer = (
                '<div class="action-footer">'
                f'<span class="action-pill">细节 {detail}</span>'
                f'<span class="action-pill">计划 {goal}</span>'
                f'<span class="action-pill">规划坐标 {planned_target}</span>'
                '</div>'
            )

        cards.append(
            (
                f'<article class="{card_class}">'
                '<div class="action-card-top">'
                f'<div class="action-card-agent"><span class="action-card-dot" style="background:{color};"></span>{agent}</div>'
                '<div class="action-card-meta">'
                f'<span class="{pill_class}">{action}</span>'
                f'<span class="action-pill">来源 {source}</span>'
                '</div>'
                '</div>'
                f'<div class="action-agent-profile">Family {family} · {trait} · {persona}</div>'
                f'<div class="action-summary">{detail}</div>'
                f'{message_box}'
                f'<div class="action-footer"><span class="action-pill">说明 {note}</span><span class="action-pill">社会 {social}</span></div>'
                f'{footer}'
                '</article>'
            )
        )

    st.markdown(f'<div class="action-stack">{"".join(cards)}</div></div>', unsafe_allow_html=True)


def _render_map(world: WorldState, moved_agent_ids: set[str] | None = None, title: str = "地图") -> None:
    moved_agent_ids = moved_agent_ids or set()
    cells: list[str] = []
    agent_lookup: dict[Position, tuple[str, str]] = {}
    for agent_id, agent in world.agents.items():
        if not agent.alive:
            continue
        marker = agent_id.split("_")[-1][0].upper()
        if agent.position in agent_lookup:
            marker = "*"
        agent_lookup[agent.position] = (marker, agent_id)

    for y_coord in range(world.height):
        for x_coord in range(world.width):
            position = Position(x_coord, y_coord)
            label = "."
            class_name = "map-cell map-cell-empty"
            inline_style = ""
            if position in world.resources:
                resource = world.resources[position]
                if resource.kind is ResourceType.GOLD:
                    label = "G"
                    class_name = "map-cell map-cell-gold"
                else:
                    label = "F"
                    class_name = "map-cell map-cell-food"
            if position in agent_lookup:
                label, agent_id = agent_lookup[position]
                if label == "*":
                    class_name = "map-cell map-cell-overlap"
                else:
                    inline_style = f"background:{AGENT_COLORS.get(agent_id, '#444')};"
                    class_name = "map-cell map-cell-agent"
                    if agent_id in moved_agent_ids:
                        class_name += " map-cell-moved"
            cells.append(
                f'<div class="{class_name}" style="{inline_style}"><span class="cell-coord">{x_coord},{y_coord}</span>{escape(label)}</div>'
            )

    legend = """
        <div class="legend">
            <span class="legend-chip"><span class="legend-swatch" style="background:#ece5d8"></span>空地</span>
            <span class="legend-chip"><span class="legend-swatch" style="background:#6e9f52"></span>食物</span>
            <span class="legend-chip"><span class="legend-swatch" style="background:#d5a021"></span>金矿</span>
            <span class="legend-chip"><span class="legend-swatch" style="background:#c84c2f"></span>Agent A</span>
            <span class="legend-chip"><span class="legend-swatch" style="background:#1f6b5a"></span>Agent B</span>
            <span class="legend-chip"><span class="legend-swatch" style="background:#1f4e79"></span>Agent C</span>
            <span class="legend-chip"><span class="legend-swatch" style="background:#ff8e3d"></span>本步有移动</span>
        </div>
    """
    st.markdown(
        (
            f'<div class="panel-shell"><div class="section-heading">{title}</div>{legend}'
            f'<div class="map-wrap" style="grid-template-columns: repeat({world.width}, minmax(34px, 1fr));">'
            f'{"".join(cells)}</div></div>'
        ),
        unsafe_allow_html=True,
    )


def _render_step_bar(step_title: str, step_value: str, alive_count: int, action_count: int) -> None:
    st.markdown(
        (
            '<div class="step-bar">'
            f'<div class="step-card"><div class="step-label">步骤</div><div class="step-value">{step_title}</div></div>'
            f'<div class="step-card"><div class="step-label">存活 Agent</div><div class="step-value">{alive_count}</div></div>'
            f'<div class="step-card"><div class="step-label">本步动作数</div><div class="step-value">{action_count}</div></div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _build_moved_agent_ids(current_world: WorldState, previous_world: WorldState | None) -> set[str]:
    if previous_world is None:
        return set()
    moved_agent_ids: set[str] = set()
    for agent_id, agent in current_world.agents.items():
        previous_agent = previous_world.agents.get(agent_id)
        if previous_agent is not None and previous_agent.position != agent.position:
            moved_agent_ids.add(agent_id)
    return moved_agent_ids


def _select_replay_frame(replay: dict[str, Any], replay_key: str) -> tuple[WorldState, list[dict[str, Any]], int, int]:
    snapshots = replay.get("snapshots", [])
    if not snapshots:
        fallback_world = deserialize_world_state(replay["world"]) if "world" in replay else WorldState(0, 10, 10, {}, {})
        return fallback_world, replay.get("events", []), 0, 1

    state_key = f"frame_index_{replay_key}"
    if state_key not in st.session_state or st.session_state[state_key] >= len(snapshots):
        st.session_state[state_key] = len(snapshots) - 1

    prev_col, next_col, reset_col = st.columns([1, 1, 1])
    if prev_col.button("上一步", key=f"prev_{replay_key}", width="stretch"):
        st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
    if next_col.button("下一步", key=f"next_{replay_key}", width="stretch"):
        st.session_state[state_key] = min(len(snapshots) - 1, st.session_state[state_key] + 1)
    if reset_col.button("终局", key=f"reset_{replay_key}", width="stretch"):
        st.session_state[state_key] = len(snapshots) - 1

    selected_index = st.slider(
        "复盘进度",
        min_value=0,
        max_value=len(snapshots) - 1,
        key=state_key,
        label_visibility="collapsed",
    )
    snapshot = snapshots[selected_index]
    selected_world = deserialize_world_state(snapshot["world"])
    selected_events = replay.get("events", [])[: int(snapshot.get("event_count", 0))]
    return selected_world, selected_events, selected_index, len(snapshots)


def _render_workspace(
    world: WorldState,
    *,
    step_title: str,
    action_rows: list[dict[str, str]],
    moved_agent_ids: set[str],
    map_title: str,
) -> None:
    _render_step_bar(step_title, str(world.turn), sum(1 for agent in world.agents.values() if agent.alive), len(action_rows))
    left, right = st.columns([1.45, 0.95])
    with left:
        _render_map(world, moved_agent_ids=moved_agent_ids, title=map_title)
    with right:
        _render_action_panel(action_rows, title="当前步行动")


def _run_live_simulation(
    mode: str,
    render_delay_seconds: float,
    map_size: int,
    max_turns: int,
    stop_when_one_agent_remains: bool,
) -> dict[str, Any]:
    config = default_config()
    config.width = map_size
    config.height = map_size
    config.max_turns = max_turns
    config.stop_when_one_agent_remains = stop_when_one_agent_remains
    config.render_interval_seconds = render_delay_seconds
    if mode == "random":
        world, agents = build_random_simulation(config)
        replay_name = "streamlit_random_run.json"
    elif mode == "heuristic":
        world, agents = build_heuristic_simulation(config)
        replay_name = "streamlit_heuristic_run.json"
    elif mode == "llm":
        world, agents = build_llm_simulation(config)
        replay_name = "streamlit_llm_run.json"
    else:
        world, agents = build_scripted_simulation(config)
        replay_name = "streamlit_scripted_run.json"

    workspace_placeholder = st.empty()
    status_placeholder = st.empty()
    scheduler = Scheduler(config)
    last_world: WorldState | None = None

    with workspace_placeholder.container():
        _render_workspace(
            world,
            step_title="准备中",
            action_rows=[],
            moved_agent_ids=set(),
            map_title="实时地图",
        )
    if mode == "llm":
        status_placeholder.info("LLM 模式已启动，正在等待模型生成首回合决策。首回合返回前动作面板不会刷新，这是正常现象。")
    else:
        status_placeholder.info("模拟已启动，正在生成首回合动作。")

    def on_turn(current_world: WorldState, _event_bus: Any, current_actions: Any) -> None:
        nonlocal last_world
        moved_agent_ids = _build_moved_agent_ids(current_world, last_world)
        action_rows = _build_live_action_rows(current_actions, agents)
        with workspace_placeholder.container():
            _render_workspace(
                current_world,
                step_title="模拟模式",
                action_rows=action_rows,
                moved_agent_ids=moved_agent_ids,
                map_title="实时地图",
            )
        status_placeholder.info(f"已完成第 {current_world.turn} 回合，正在准备下一回合动作。")
        if config.render_interval_seconds > 0:
            time.sleep(config.render_interval_seconds)
        last_world = deserialize_world_state({
            "turn": current_world.turn,
            "width": current_world.width,
            "height": current_world.height,
            "agents": {
                agent_id: {
                    "agent_id": agent.agent_id,
                    "family_id": agent.family_id,
                    "energy": agent.energy,
                    "alive": agent.alive,
                    "position": {"x": agent.position.x, "y": agent.position.y},
                    "inventory": dict(agent.inventory),
                }
                for agent_id, agent in current_world.agents.items()
            },
            "resources": [
                {
                    "position": {"x": position.x, "y": position.y},
                    "kind": resource.kind.value,
                    "amount": resource.amount,
                }
                for position, resource in current_world.resources.items()
            ],
        })

    try:
        with st.spinner("模拟运行中..."):
            result = scheduler.run(world, agents, replay_name=replay_name, on_turn=on_turn)
        status_placeholder.success(
            f"模拟完成：共运行 {result.metrics.turn} 回合，最终存活 {result.metrics.alive_agents} 个 agent。"
        )
        return load_replay(result.replay_path)
    except Exception as error:
        status_placeholder.error(f"模拟启动失败：{error}")
        raise


def _render_replay_mode(catalog: list[dict[str, Any]]) -> None:
    if not catalog:
        st.info("当前没有可复盘的记录，请先运行一次模拟。")
        return

    replay_lookup = {entry["label"]: entry for entry in catalog}
    selected_label = st.selectbox("选择回放", options=list(replay_lookup), index=len(replay_lookup) - 1)
    replay = load_replay(replay_lookup[selected_label]["path"])
    replay_key = str(replay.get("metadata", {}).get("run_id", selected_label))
    current_world, current_events, frame_index, total_frames = _select_replay_frame(replay, replay_key)

    previous_world = None
    if replay.get("snapshots") and frame_index > 0:
        previous_world = deserialize_world_state(replay["snapshots"][frame_index - 1]["world"])
    moved_agent_ids = _build_moved_agent_ids(current_world, previous_world)
    action_rows = _build_replay_action_rows(current_events, current_world.turn, replay.get("metadata", {}), current_world)

    _render_workspace(
        current_world,
        step_title=f"复盘模式 {frame_index + 1}/{total_frames}",
        action_rows=action_rows,
        moved_agent_ids=moved_agent_ids,
        map_title="复盘地图",
    )


def _render_metric_chart(title: str, rows: list[dict[str, float | int]], metric_name: str, description: str) -> None:
    st.markdown(
        f'<div class="panel-shell"><div class="section-heading">{title}</div><div class="section-copy">{description}</div>',
        unsafe_allow_html=True,
    )
    if not rows:
        st.info("当前实验还没有可展示的代际汇总数据。")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    chart_frame = pd.DataFrame(rows)
    st.line_chart(chart_frame, x="generation", y=metric_name, height=260)
    st.dataframe(chart_frame, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_evolution_dashboard(experiment_catalog: list[dict[str, Any]]) -> None:
    st.markdown(
        '<div class="panel-shell"><div class="section-heading">代际演化大盘</div><div class="section-copy">读取 experiment_manifest.json，展示跨代际的生存、通信熵和欺骗频率趋势。</div></div>',
        unsafe_allow_html=True,
    )
    if not experiment_catalog:
        st.info("当前 artifacts 下还没有 experiment_manifest.json。请先运行一次多代实验。")
        return

    experiment_lookup = {entry["label"]: entry for entry in experiment_catalog}
    selected_label = st.selectbox("选择实验", options=list(experiment_lookup), index=len(experiment_lookup) - 1)
    selected_experiment = experiment_lookup[selected_label]
    manifest = selected_experiment["manifest"]
    generation_summaries = selected_experiment["generation_summaries"]
    latest_summary = selected_experiment.get("latest_summary", {})

    summary_cols = st.columns(4)
    summary_cols[0].metric("实验 ID", selected_experiment["experiment_id"])
    summary_cols[1].metric("Run Group", selected_experiment["run_group"])
    summary_cols[2].metric("总代数", str(selected_experiment["generations"]))
    summary_cols[3].metric("每代运行数", str(selected_experiment["runs_per_generation"]))

    highlight_cols = st.columns(3)
    highlight_cols[0].metric("最新平均存活回合", f"{float(latest_summary.get('average_survival_turn', 0.0)):.2f}")
    highlight_cols[1].metric("最新信息熵", f"{float(latest_summary.get('entropy', 0.0)):.4f}")
    highlight_cols[2].metric("最新欺骗频率", f"{float(latest_summary.get('deception_frequency', 0.0)):.4f}")

    left, right = st.columns(2)
    with left:
        _render_metric_chart(
            "平均存活回合趋势",
            build_generation_metric_rows(manifest, "average_survival_turn"),
            "average_survival_turn",
            "X 轴为代数，Y 轴为平均存活回合数，用于观察策略是否随代际改进。",
        )
        _render_metric_chart(
            "欺骗频率趋势",
            build_generation_metric_rows(manifest, "deception_frequency"),
            "deception_frequency",
            "X 轴为代数，Y 轴为 deception frequency，用于观察高阶博弈是否涌现。",
        )
    with right:
        _render_metric_chart(
            "语言熵趋势",
            build_generation_metric_rows(manifest, "entropy"),
            "entropy",
            "X 轴为代数，Y 轴为香农信息熵，用于观察消息系统是否自发压缩。",
        )
        st.markdown(
            '<div class="panel-shell"><div class="section-heading">代际汇总表</div><div class="section-copy">generation_summaries 原始聚合结果，方便导出到汇报材料。</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(generation_summaries), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Project Darwin", layout="wide")
    _inject_styles()
    _render_header()

    artifact_root = Path("artifacts")
    catalog = build_replay_catalog(artifact_root)
    experiment_catalog = build_experiment_catalog(artifact_root)
    llm_adapter = LLMAdapter()

    with st.sidebar:
        st.header("工作模式")
        workspace_mode = st.radio("模式", options=["模拟模式", "复盘模式"], index=0)
        st.divider()
        if workspace_mode == "模拟模式":
            config_defaults = default_config()
            st.header("模拟控制")
            simulation_mode = st.selectbox("Agent 模式", options=["scripted", "random", "heuristic", "llm"], index=2)
            map_size = st.slider("地图边长", min_value=5, max_value=12, value=config_defaults.width, step=1)
            max_turns = st.slider("最大轮次", min_value=1, max_value=200, value=config_defaults.max_turns, step=1)
            stop_when_one_agent_remains = st.checkbox(
                "只剩 1 个 agent 存活时提前结束",
                value=config_defaults.stop_when_one_agent_remains,
            )
            render_delay_ms = st.slider("步进延迟（毫秒）", min_value=0, max_value=1200, value=220, step=20)
            run_clicked = st.button("开始模拟", width="stretch")
            st.caption(f"当前地图固定为方形：{map_size} x {map_size}")
            if simulation_mode == "llm":
                if llm_adapter.is_configured():
                    st.success(f"LLM 已配置：{llm_adapter.model_name}")
                else:
                    st.error("LLM 未配置，当前会自动退回 heuristic fallback。")
        else:
            run_clicked = False
            simulation_mode = "heuristic"
            map_size = 0
            max_turns = 0
            stop_when_one_agent_remains = True
            render_delay_ms = 0

    tab_workspace, tab_evolution = st.tabs(["单局工作台", "代际演化大盘"])

    with tab_workspace:
        if workspace_mode == "模拟模式":
            if run_clicked:
                _run_live_simulation(
                    simulation_mode,
                    render_delay_ms / 1000.0,
                    map_size,
                    max_turns,
                    stop_when_one_agent_remains,
                )
            else:
                st.info("点击左侧“开始模拟”后，地图会逐步显示每一回合 agent 的行动。")
        else:
            _render_replay_mode(catalog)

    with tab_evolution:
        _render_evolution_dashboard(experiment_catalog)


if __name__ == "__main__":
    main()