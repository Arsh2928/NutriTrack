"""Agent 3: Goal and Calorie Budget Agent — manages targets, remaining allowances, and meal fit checks."""

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

AGENT_KEY = "goal_budget"
AGENT_NAME = "Goal and Budget Agent"
AGENT_ROLE = "Manages calorie and protein targets, computes remaining daily budget, and validates whether proposed meals fit."

SYSTEM_INSTRUCTIONS = """[AGENT:goal_budget]
You are the Goal and Budget Agent in NutriTrack, built on Microsoft Agent Framework.
Your responsibility is managing user nutrition targets and checking calorie/protein budgets.

Strict Rules:
1. Update targets ONLY using `set_calorie_target` and `set_protein_target` when explicitly requested by the user.
2. Always use `calculate_remaining_targets` to compute remaining calories and protein allowance accurately in Python.
3. When the user asks if a food or meal fits their budget, evaluate it using `check_meal_against_budget`.
4. NEVER prescribe medical or clinical nutrition targets on your own; use user-supplied values or clearly labeled default demonstration targets.
5. NEVER perform arithmetic in text — cite exact tool output values only.
"""

SPEC = AgentSpec(
    key=AGENT_KEY,
    name=AGENT_NAME,
    role=AGENT_ROLE,
    instructions=SYSTEM_INSTRUCTIONS,
)


def create_goal_budget_agent(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
) -> Agent:
    """Create the Microsoft Agent Framework `Agent` instance for Goal and Calorie Budget."""
    resolved_client = client if client is not None else create_chat_client()[0]
    return build_maf_agent(SPEC, ctx, resolved_client)


def run_goal_budget(
    ctx: ToolContext,
    task: str,
    *,
    client: Optional[BaseChatClient] = None,
    session: Optional[AgentSession] = None,
) -> dict[str, Any]:
    """Run the Goal and Budget Agent on a task and return structured execution output."""
    agent = create_goal_budget_agent(ctx, client=client)
    return execute_agent(agent, AGENT_KEY, ctx, task, session=session)


def make_goal_budget_runner(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
) -> Callable[[str], dict[str, Any]]:
    """Build a synchronous specialist runner registered with the Nutrition Manager."""
    resolved_client = client if client is not None else create_chat_client()[0]
    return make_specialist_runner(SPEC, ctx, resolved_client)
