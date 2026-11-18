"""
llm_agent.py
------------
The REAL plan-act loop: an LLM (via GitHub Models) decides which tool to
call, reads the tool's result, and decides what to do next — this file
is the actual "agent" in the agentic-AI sense the course asks for.

This is deliberately a thin layer. All actual work — nutrition lookup,
scaling, memory, persistence, safety notes — still lives in tools.py /
memory.py / daily_store.py, completely unchanged. This file's only job
is: give the model the user's message + the tool list, let IT decide
what to call, execute what it asks for, feed the result back, and repeat
until it has enough information to answer in plain language.

Contrast with agent.py (MealCaloricAgent):
    agent.py performs the SAME kind of multi-step, tool-driven, memory-
    backed behaviour, but decides which tool to call using hand-written
    regex patterns. It's a legitimate rule-based agent and still meets
    the letter of the CA1 rubric (tools + multi-step + memory) — but the
    decision-making itself is deterministic Python, not a model.
    LLMMealAgent replaces ONLY that decision-making step with a real LLM
    performing function-calling, which is what "Agentic AI" usually
    means in the stricter sense, and matches the course's own suggested
    GitHub Models / Microsoft Foundry lane.

    app.py uses LLMMealAgent automatically when a GITHUB_TOKEN is
    configured, and falls back to MealCaloricAgent otherwise, so the
    project runs either way.
"""

from __future__ import annotations
import json
from typing import Optional

from openai import OpenAI

from . import tools as T
from . import daily_store
from . import llm_config
from .tool_schemas import TOOL_SCHEMAS
from ..state.run_log import sanitize_error_text


SYSTEM_PROMPT = """You are the T17 Meal Calorie Helper — an informational \
meal-tracking agent. You are NOT a nutritionist or doctor.

Your job each turn:
1. Read what the user wants.
2. Decide which tool(s) to call, if any, to get the facts you need — \
never guess calorie/macro numbers yourself, always look them up.
3. Read each tool's result before deciding your next step. If the first \
tool's result changes what you should do next (e.g. a food wasn't \
found), adapt — don't blindly continue a fixed plan.
4. Once you have what you need, reply to the user in a short, clear, \
friendly message. Include the concrete numbers the tools gave you \
(calories, remaining budget, macros) — don't just say "added it".

Rules:
- Always use tools for anything numeric or food-related. Never invent \
calorie or macro values yourself.
- If a tool returns can_add_manually or a not_found status, tell the \
user plainly and suggest they can add the food themselves — do not \
pretend it succeeded.
- Keep replies concise: a few lines, not an essay.
- This is informational only — never present figures as medically exact, \
and never give medical advice.
- If the user's request needs no tool (e.g. "hi", "what can you do?"), \
just reply directly without forcing a tool call.
"""


# Maps each tool schema's function name to the real Python function that
# implements it. This is the ONLY place tool names are wired to code —
# everything else (which tool, when, in what order) is the model's call.
TOOL_DISPATCH = {
    "set_budget": lambda args: T.set_budget(args["budget_kcal"]),
    "lookup_calories": lambda args: T.lookup_calories(args["query"]),
    "add_meal": lambda args: T.add_meal(args["query"]),
    "check_snack_fit": lambda args: T.check_snack_fit(args["query"]),
    "get_daily_summary": lambda args: T.get_daily_summary(),
    "delete_meal": lambda args: T.delete_meal(
        index=args.get("index"), food_name=args.get("food_name")
    ),
    "add_custom_food_tool": lambda args: T.add_custom_food_tool(
        name=args["name"],
        calories=args["calories"],
        protein=args["protein"],
        carbs=args["carbs"],
        fat=args["fat"],
        serving_size=args.get("serving_size", "1 serving"),
    ),
    "save_last_lookup": lambda args: T.save_last_lookup(),
    "get_daily_history": lambda args: T.get_daily_history(
        limit=args.get("limit", 7)
    ),
}


class LLMMealAgent:
    """
    A real plan-act agent: an LLM decides which tools to call and in
    what order, based on the user's message and each tool's result.

    Keeps a running conversation history so the model itself also has
    turn-to-turn context (on top of the persistent SessionMemory/
    daily_store state that the tools read and write).
    """

    def __init__(self, verbose: bool = True) -> None:
        if not llm_config.has_token():
            raise RuntimeError(
                "No GITHUB_TOKEN configured. LLMMealAgent needs a free "
                "GitHub Models token — see llm_config.py for setup steps. "
                "Use agent.MealCaloricAgent instead if you don't have one yet."
            )

        self.client = OpenAI(
            base_url=llm_config.GITHUB_MODELS_BASE_URL,
            api_key=llm_config.GITHUB_TOKEN,
        )
        self.model = llm_config.MODEL_NAME
        self.verbose = verbose
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._turn = 0

    # ------------------------------------------------------------------

    def _log(self, line: str) -> None:
        if self.verbose:
            print(line)

    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Process one user turn: the LLM may call zero or more tools
        (deciding each next step from the previous tool's result) before
        producing a final natural-language reply.

        Returns the final reply text. Also prints the plan-act trace if
        verbose=True.
        """
        return self.chat_capture(user_message)["response"]

    def chat_capture(self, user_message: str) -> dict:
        """
        Same as chat(), but also returns the plan-act trace as structured
        text, for the web frontend (app.py) — mirrors MealCaloricAgent's
        chat_capture() so app.py can use either agent interchangeably.
        """
        self._turn += 1
        trace_lines: list[str] = []

        def log(line: str) -> None:
            trace_lines.append(line)
            self._log(line)

        # --- Day persistence / rollover (same mechanism as agent.py) ----
        mem = T.get_memory()
        rollover = daily_store.load_or_rollover(mem)
        if rollover["action"] == "rolled_over":
            log(
                f"🌅 New day detected — {rollover['previous_date']}'s log "
                f"has been saved to history. Starting a fresh day."
            )
        elif rollover["action"] == "restored":
            log(f"🔁 Restored today's earlier meals from disk ({len(mem.meals)} logged so far).")

        log(f"\n{'═'*60}\n👤 USER (Turn {self._turn}): {user_message}\n{'═'*60}")

        self.messages.append({"role": "user", "content": user_message})

        final_text = ""
        for round_num in range(1, llm_config.MAX_TOOL_ROUNDS + 1):
            log(f"[PLAN] Calling the model (round {round_num}) — letting it "
                f"decide the next step from everything so far…")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except Exception as exc:
                safe_err = sanitize_error_text(exc)
                final_text = (
                    f"⚠️ Could not reach the LLM ({safe_err}). Check your "
                    f"GITHUB_TOKEN and internet connection."
                )
                log(f"[ERROR] {safe_err}")
                break

            choice = response.choices[0].message

            if not choice.tool_calls:
                # Model decided it has enough information — final answer.
                final_text = choice.content or ""
                self.messages.append({"role": "assistant", "content": final_text})
                log("[PLAN] Model decided no further tools are needed — final answer ready.")
                break

            # Model asked for one or more tool calls this round.
            self.messages.append({
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [tc.model_dump() for tc in choice.tool_calls],
            })

            for tc in choice.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                log(f"[ACT] Model chose tool: {name}({args})")

                if name not in TOOL_DISPATCH:
                    result = {"status": "error", "error": f"Unknown tool '{name}'."}
                else:
                    try:
                        result = TOOL_DISPATCH[name](args)
                    except Exception as exc:
                        result = {"status": "error", "error": str(exc)}

                log(f"[TOOL RESULT] {json.dumps(result, default=str)[:300]}")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

            log("[OBSERVE] Model will read the tool result(s) above and "
                "decide the next step itself.")
        else:
            final_text = (
                "⚠️ Reached the maximum number of tool-call steps for this "
                "turn without a final answer. Try rephrasing your request."
            )

        if self.verbose:
            print(f"\n🤖 RESPONSE:\n{final_text}")

        # Persist today's state after the turn, same as the rule-based agent.
        daily_store.sync_after_change(T.get_memory())

        trace_text = "\n".join(trace_lines).strip()
        return {
            "response": final_text.strip(),
            "trace": trace_text,
            "raw": trace_text + f"\n\n🤖 RESPONSE:\n{final_text}",
        }
