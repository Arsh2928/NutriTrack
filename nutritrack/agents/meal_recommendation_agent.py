"""Agent 4: Meal Recommendation Agent — filters, estimates, ranks, and validates next-meal suggestions."""

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

AGENT_KEY = "meal_recommendation"
AGENT_NAME = "Meal Recommendation Agent"
AGENT_ROLE = "Suggests, scores, and validates meal combinations fitting remaining calories, protein, and preferences."

SYSTEM_INSTRUCTIONS = """[AGENT:meal_recommendation]
You are the Meal Recommendation Agent in NutriTrack, built on Microsoft Agent Framework.
Your responsibility is recommending balanced next meals grounded strictly in the food database and remaining targets.

Strict Rules:
1. Call `get_recommendation_candidates` first, passing any user dietary preferences or restrictions (`diet`, `avoid`, `prefer`).
2. Call `estimate_meal_nutrition` on the returned `candidate_ids` to compute real calories, protein, carbs, and fat from the database.
3. Call `rank_meal_candidates` to score and sort options against remaining calories and remaining protein.
4. Call `validate_recommendation_fit` on `best_id` to confirm the top suggestion fits the user's remaining budget.
5. Report recommended foods, portions, macros, why the recommendation fits, suitable alternatives, and the non-medical estimation disclaimer directly from tool outputs. Never invent foods or nutrition numbers.
"""

SPEC = AgentSpec(
    key=AGENT_KEY,
    name=AGENT_NAME,
    role=AGENT_ROLE,
    instructions=SYSTEM_INSTRUCTIONS,
)


def create_meal_recommendation_agent(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
) -> Agent:
    """Create the Microsoft Agent Framework `Agent` instance for Meal Recommendation."""
    resolved_client = client if client is not None else create_chat_client()[0]
    return build_maf_agent(SPEC, ctx, resolved_client)


def run_meal_recommendation(
    ctx: ToolContext,
    task: str,
    *,
    client: Optional[BaseChatClient] = None,
    session: Optional[AgentSession] = None,
) -> dict[str, Any]:
    """Run the Meal Recommendation Agent on a task and return structured execution output."""
    agent = create_meal_recommendation_agent(ctx, client=client)
    return execute_agent(agent, AGENT_KEY, ctx, task, session=session)


def make_meal_recommendation_runner(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
) -> Callable[[str], dict[str, Any]]:
    """Build a synchronous specialist runner registered with the Nutrition Manager."""
    resolved_client = client if client is not None else create_chat_client()[0]
    return make_specialist_runner(SPEC, ctx, resolved_client)
