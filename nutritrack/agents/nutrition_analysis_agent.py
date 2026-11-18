"""Agent 2: Nutrition Analysis Agent — computes daily totals, macro splits, weekly trends, and target comparisons."""

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

AGENT_KEY = "nutrition_analysis"
AGENT_NAME = "Nutrition Analysis Agent"
AGENT_ROLE = "Calculates daily nutrition totals, macro percentages, 7-day trends, unusual days, and target comparisons."

SYSTEM_INSTRUCTIONS = """[AGENT:nutrition_analysis]
You are the Nutrition Analysis Agent in NutriTrack, built on Microsoft Agent Framework.
Your responsibility is analyzing logged nutritional intake from the shared NutritionState.

Strict Rules:
1. Use `calculate_daily_totals` to obtain exact daily calories, protein, carbs, fat, and meal counts.
2. Use `calculate_macro_split` to retrieve protein, carbs, and fat gram breakdown and caloric share percentages.
3. Use `get_weekly_trends` for multi-day or 7-day average intake, calorie target compliance, and unusual intake days.
4. Use `compare_intake_with_targets` to compare daily calories and protein against the user's targets.
5. NEVER perform mental math or fabricate missing days. If fewer than 7 days of records exist, state the exact recorded period and missing-day basis reported by `get_weekly_trends`.
6. Report all numbers strictly as returned by your Python tools.
"""

SPEC = AgentSpec(
    key=AGENT_KEY,
    name=AGENT_NAME,
    role=AGENT_ROLE,
    instructions=SYSTEM_INSTRUCTIONS,
)


def create_nutrition_analysis_agent(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
) -> Agent:
    """Create the Microsoft Agent Framework `Agent` instance for Nutrition Analysis."""
    resolved_client = client if client is not None else create_chat_client()[0]
    return build_maf_agent(SPEC, ctx, resolved_client)


def run_nutrition_analysis(
    ctx: ToolContext,
    task: str,
    *,
    client: Optional[BaseChatClient] = None,
    session: Optional[AgentSession] = None,
) -> dict[str, Any]:
    """Run the Nutrition Analysis Agent on a task and return structured execution output."""
    agent = create_nutrition_analysis_agent(ctx, client=client)
    return execute_agent(agent, AGENT_KEY, ctx, task, session=session)


def make_nutrition_analysis_runner(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
) -> Callable[[str], dict[str, Any]]:
    """Build a synchronous specialist runner registered with the Nutrition Manager."""
    resolved_client = client if client is not None else create_chat_client()[0]
    return make_specialist_runner(SPEC, ctx, resolved_client)
