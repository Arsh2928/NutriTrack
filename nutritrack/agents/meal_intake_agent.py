"""Agent 1: Meal Intake Agent — parses food entries, resolves nutrition, and updates shared state."""

from __future__ import annotations

from typing import Any, Callable, Optional

from agent_framework import Agent, AgentSession, BaseChatClient

from nutritrack.agents.base import (
    AgentSpec,
    build_maf_agent,
    execute_agent,
    make_specialist_runner,
)
from nutritrack.agents.runtime import create_chat_client
from nutritrack.tools.base import ToolContext

AGENT_KEY = "meal_intake"
AGENT_NAME = "Meal Intake Agent"
AGENT_ROLE = "Parses natural-language food logs, validates quantities, looks up nutrition, and logs or removes meals."

SYSTEM_INSTRUCTIONS = """[AGENT:meal_intake]
You are the Meal Intake Agent in NutriTrack, built on Microsoft Agent Framework.
Your sole responsibility is managing meal entries and looking up verified nutrition data.

Strict Rules:
1. Always call `parse_meal_entry` to split user text into individual food items and quantities.
2. Call `lookup_food_nutrition` for parsed food items to retrieve verified calories and macros.
3. Call `add_meal_to_state` to validate quantities, check duplicate meals, and save logged foods to the shared NutritionState.
4. Use `remove_meal_from_state` when the user asks to delete, remove, or undo an entry.
5. Only call `register_custom_food` when the user explicitly supplies the nutrition facts (calories, protein, carbs, fat).
6. NEVER invent or estimate nutritional values for foods missing from the database. If an item is unknown, ambiguous, or missing a quantity, preserve the tool's follow-up question clearly for the user.
7. Summarize strictly from the JSON returned by your Python tools.
"""

SPEC = AgentSpec(
    key=AGENT_KEY,
    name=AGENT_NAME,
    role=AGENT_ROLE,
    instructions=SYSTEM_INSTRUCTIONS,
)


def create_meal_intake_agent(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
) -> Agent:
    """Create the Microsoft Agent Framework `Agent` instance for Meal Intake."""
    resolved_client = client if client is not None else create_chat_client()[0]
    return build_maf_agent(SPEC, ctx, resolved_client)


def run_meal_intake(
    ctx: ToolContext,
    task: str,
    *,
    client: Optional[BaseChatClient] = None,
    session: Optional[AgentSession] = None,
) -> dict[str, Any]:
    """Run the Meal Intake Agent on a task and return structured execution output."""
    agent = create_meal_intake_agent(ctx, client=client)
    return execute_agent(agent, AGENT_KEY, ctx, task, session=session)


def make_meal_intake_runner(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
) -> Callable[[str], dict[str, Any]]:
    """Build a synchronous specialist runner registered with the Nutrition Manager."""
    resolved_client = client if client is not None else create_chat_client()[0]
    return make_specialist_runner(SPEC, ctx, resolved_client)
