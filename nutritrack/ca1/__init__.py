"""CA1 Meal Calorie Helper -- preserved as a sub-package.

Contents: rule-based agent (agent.py), optional LLM agent (llm_agent.py),
nine tools (tools.py), session memory (memory.py), day persistence
(daily_store.py), food database (food_database.py).

The only differences from the CA1 originals are import statements (flat
`import tools` became package-relative) and the default storage location
(see ../paths.py). See docs/MIGRATION_MAP.md for the exact list.

The rule-based terminal chat can be started with:
    python -m nutritrack.ca1.agent
"""
