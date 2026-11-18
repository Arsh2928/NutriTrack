"""
llm_config.py
-------------
Configuration for the LLM planning layer (llm_agent.py).

This project uses GitHub Models — free, OpenAI-SDK-compatible model
access tied to your GitHub account, no billing required. This is the
"lane" your course brief calls out explicitly (alongside Microsoft
Foundry).

How to get a free token:
    1. Sign in to https://github.com
    2. Go to Settings → Developer settings → Personal access tokens
       → Fine-grained tokens → "Generate new token"
       (No repository access is needed — this token is only used to
       authenticate to the Models API, not to touch any repo.)
    3. Copy the token (starts with "github_pat_..." or "ghp_...").
    4. Set it as an environment variable:

       Windows (cmd):        set GITHUB_TOKEN=your_token_here
       Windows (PowerShell):  $env:GITHUB_TOKEN="your_token_here"
       macOS / Linux:         export GITHUB_TOKEN=your_token_here

    You can also browse available models and test them first at
    https://github.com/marketplace/models

If no token is set, llm_agent.py cannot run — app.py detects this and
automatically falls back to the deterministic rule-based agent
(agent.py) instead, so the project still works with zero setup. Adding
a token upgrades it to a real LLM-driven agent without changing anything
else.
"""

import os

# Legacy CA1 GitHub Models configuration is decommissioned under strict 3-mode AI isolation
# (Gemini, Groq, Offline only). Do not read or enable GITHUB_TOKEN automatically.
GITHUB_TOKEN = ""
GITHUB_MODELS_BASE_URL = ""
MODEL_NAME = ""
MAX_TOOL_ROUNDS = 6


def has_token() -> bool:
    """Decommissioned: legacy GitHub Models provider is disabled."""
    return False
