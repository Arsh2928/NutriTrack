"""Deterministic Python tools for the five agents.

    TOOLS_BY_AGENT[agent_key]      the tool functions of each agent (each takes `ctx` first)
    tools_for(agent_key, ctx)      the same tools bound to a ToolContext -- what an agent is given
    new_context(...)               build a ToolContext for one run
"""
from ..state import RunLog, StateManager, new_run_id
from . import analysis_tools, budget_tools, intake_tools, manager_tools, recommendation_tools
from .base import AGENTS, SPECIALISTS, LLMUnavailableError, ToolContext, agent_tool, bind, fail, ok

TOOLS_BY_AGENT = {
    "meal_intake": [intake_tools.parse_meal_entry, intake_tools.lookup_food_nutrition,
                    intake_tools.add_meal_to_state, intake_tools.remove_meal_from_state,
                    intake_tools.register_custom_food],
    "nutrition_analysis": [analysis_tools.calculate_daily_totals, analysis_tools.calculate_macro_split,
                           analysis_tools.get_weekly_trends, analysis_tools.compare_intake_with_targets],
    "goal_budget": [budget_tools.set_calorie_target, budget_tools.set_protein_target,
                    budget_tools.calculate_remaining_targets, budget_tools.check_meal_against_budget],
    "meal_recommendation": [recommendation_tools.get_recommendation_candidates,
                            recommendation_tools.estimate_meal_nutrition,
                            recommendation_tools.rank_meal_candidates,
                            recommendation_tools.validate_recommendation_fit],
    "nutrition_manager": [manager_tools.read_shared_state, manager_tools.delegate_to_specialist,
                          manager_tools.validate_agent_results, manager_tools.update_workflow_status,
                          manager_tools.append_run_log],
}


def tools_for(agent_key, ctx):
    return [bind(tool, ctx) for tool in TOOLS_BY_AGENT[agent_key]]


def new_context(state_manager, run_log, user_id="demo_user", task="", specialists=None, run_id=None):
    return ToolContext(state_manager=state_manager, run_log=run_log, user_id=user_id, task=task,
                       run_id=run_id or new_run_id(), specialists=specialists if specialists is not None else {})


__all__ = ["TOOLS_BY_AGENT", "tools_for", "new_context", "ToolContext", "bind", "AGENTS", "SPECIALISTS",
           "LLMUnavailableError", "agent_tool", "ok", "fail", "RunLog", "StateManager"]
