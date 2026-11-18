"""Five Microsoft Agent Framework specialist and manager agents for NutriTrack."""

from nutritrack.agents.base import (
    AgentSpec,
    build_maf_agent,
    build_maf_tools,
    execute_agent,
    extract_tool_trace,
    make_specialist_runner,
)
from nutritrack.agents.goal_budget_agent import (
    create_goal_budget_agent,
    make_goal_budget_runner,
    run_goal_budget,
)
from nutritrack.agents.meal_intake_agent import (
    create_meal_intake_agent,
    make_meal_intake_runner,
    run_meal_intake,
)
from nutritrack.agents.meal_recommendation_agent import (
    create_meal_recommendation_agent,
    make_meal_recommendation_runner,
    run_meal_recommendation,
)
from nutritrack.agents.nutrition_analysis_agent import (
    create_nutrition_analysis_agent,
    make_nutrition_analysis_runner,
    run_nutrition_analysis,
)
from nutritrack.agents.nutrition_manager_agent import (
    NutriTrackOrchestrator,
    create_nutrition_manager_agent,
    register_specialists,
    run_nutritrack_turn,
)
from nutritrack.agents.runtime import (
    DeterministicNutriTrackChatClient,
    FailingChatClient,
    ScriptedChatClient,
    create_chat_client,
    detect_llm_config,
    load_env,
    plan_specialists_for_request,
    run_async,
    text_response,
    tool_call_response,
)

__all__ = [
    "AgentSpec",
    "DeterministicNutriTrackChatClient",
    "FailingChatClient",
    "NutriTrackOrchestrator",
    "ScriptedChatClient",
    "build_maf_agent",
    "build_maf_tools",
    "create_chat_client",
    "create_goal_budget_agent",
    "create_meal_intake_agent",
    "create_meal_recommendation_agent",
    "create_nutrition_analysis_agent",
    "create_nutrition_manager_agent",
    "detect_llm_config",
    "execute_agent",
    "extract_tool_trace",
    "load_env",
    "make_goal_budget_runner",
    "make_meal_intake_runner",
    "make_meal_recommendation_runner",
    "make_nutrition_analysis_runner",
    "make_specialist_runner",
    "plan_specialists_for_request",
    "register_specialists",
    "run_async",
    "run_goal_budget",
    "run_meal_intake",
    "run_meal_recommendation",
    "run_nutrition_analysis",
    "run_nutritrack_turn",
    "text_response",
    "tool_call_response",
]
