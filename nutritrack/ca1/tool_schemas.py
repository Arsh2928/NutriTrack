"""
tool_schemas.py
----------------
OpenAI-format function-calling schemas describing every tool in tools.py
that the LLM planning layer (llm_agent.py) is allowed to call.

These schemas are what actually make this a real agent rather than a
scripted dispatcher: the model reads each tool's name, description, and
parameters, and DECIDES for itself which one to call (if any), based on
the user's message and the results of any tools it has already called
this turn. Nothing here hard-codes which tool follows which — that
decision is made by the LLM at request time.

Each schema's "description" is written for the MODEL to read, so it is
deliberately explicit about when to use each tool — this is effectively
the model's "documentation" of its own toolbox.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "set_budget",
            "description": (
                "Set the user's daily calorie budget in kcal. Call this "
                "whenever the user states or changes their daily calorie "
                "target, e.g. 'my budget is 2200 kcal' or 'set my budget "
                "to 2000'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_kcal": {
                        "type": "number",
                        "description": "Daily calorie budget in kcal. Must be positive.",
                    }
                },
                "required": ["budget_kcal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_calories",
            "description": (
                "Look up estimated calories and macros (protein/carbs/fat) "
                "for a food WITHOUT logging it. Use this when the user is "
                "just asking about a food's nutrition, or as a first step "
                "before deciding whether to log it or whether a snack fits "
                "the remaining budget. Tries the local database first, "
                "then an online fallback if not found locally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language food description, e.g. "
                            "'3 boiled eggs', '200 g chicken breast', 'banana'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_meal",
            "description": (
                "Log a food to today's meal record and update the running "
                "daily totals (calories, protein, carbs, fat). Call this "
                "when the user says they ate/had/are adding a food — e.g. "
                "'add 3 boiled eggs', 'I had 200g chicken breast'. This "
                "tool internally looks up the food's nutrition itself, so "
                "you do not need to call lookup_calories first unless you "
                "specifically want to show the nutrition before logging."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language food + quantity, e.g. '3 boiled eggs'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_snack_fit",
            "description": (
                "Decide whether a planned snack/food fits within the "
                "user's REMAINING calorie budget, WITHOUT logging it. Use "
                "this for questions like 'can I eat a banana?' or 'do I "
                "have room for 2 slices of bread?'. Looks up the food's "
                "calories, reads the remaining budget from memory, and "
                "returns a fits/does-not-fit decision with the numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language food + quantity being considered.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_summary",
            "description": (
                "Get today's complete state: budget, all logged meals, "
                "total calories/protein/carbs/fat, remaining calories, and "
                "the macro percentage split. Call this when the user asks "
                "for a summary, totals, or 'how am I doing today'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_meal",
            "description": (
                "Remove ONE previously-logged meal from today's log and "
                "recompute totals. Provide EITHER index (the meal's "
                "1-based position — call get_daily_summary first if you "
                "don't already know it) OR food_name (matches the most "
                "recently logged meal with that name). Use when the user "
                "says 'delete', 'remove', or 'undo' a specific logged meal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based position of the meal to remove, if known.",
                    },
                    "food_name": {
                        "type": "string",
                        "description": "Food name to match against today's log, if index is not known.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_custom_food_tool",
            "description": (
                "Permanently add a NEW food the user describes themselves "
                "— a homemade dish, or a branded product not found by "
                "lookup_calories — to the local database with manually "
                "given nutrition values, so it can be found instantly next "
                "time. Only call this when the user explicitly gives you "
                "the nutrition numbers to save (calories/protein/carbs/fat), "
                "not as a first attempt at looking something up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Food name to save it under."},
                    "calories": {"type": "number", "description": "kcal for one serving."},
                    "protein": {"type": "number", "description": "Grams of protein per serving."},
                    "carbs": {"type": "number", "description": "Grams of carbs per serving."},
                    "fat": {"type": "number", "description": "Grams of fat per serving."},
                    "serving_size": {
                        "type": "string",
                        "description": (
                            "Human-readable serving description, e.g. "
                            "'100 g' (scalable by weight) or '1 bowl (250 g)' "
                            "(treated as one indivisible serving)."
                        ),
                    },
                },
                "required": ["name", "calories", "protein", "carbs", "fat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_last_lookup",
            "description": (
                "Permanently save the MOST RECENT ONLINE (API) lookup "
                "result into the local database, so it's found instantly "
                "and offline next time. Only useful right after "
                "lookup_calories or add_meal returned a result with "
                "source='api'. Call when the user says 'save this food' "
                "or similar, right after such a lookup."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_history",
            "description": (
                "Get a summary of past (fully completed) days — NOT "
                "today. Use when the user asks about yesterday, past "
                "days, or their history/trend over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many past days to return (default 7).",
                    }
                },
            },
        },
    },
]
