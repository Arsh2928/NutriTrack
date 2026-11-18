"""
config.py
---------
Configuration for the OPTIONAL external nutrition API fallback.

The agent works completely offline using the built-in food_database.py.
This config only matters for lookup_calories()'s online fallback, used
when a food is missing from the local database.

Two providers are supported — pick with NUTRITION_API_PROVIDER:

  "openfoodfacts"  (DEFAULT — no signup, no key, nothing to break)
      Free, open, crowdsourced product database. No API key required
      at all. Best default for a college project: zero setup friction.
      Coverage is strongest for PACKAGED/BRANDED products (like Pintola
      Muesli) since it's a barcode/product database, not a general
      nutrition calculator.

  "calorieninjas"  (needs a free key — now issued via the API Ninjas
                     platform, since CalorieNinjas merged into it in 2025)
      Natural-language nutrition lookups ("3 boiled eggs" understood
      directly). Get a key at:
          1. https://api-ninjas.com/register  (sign up — free, no card)
          2. Copy the API key from your dashboard
          3. Set it as shown below
      One API Ninjas account/key works across ALL their APIs, including
      the nutrition endpoint — you don't need a separate CalorieNinjas
      account anymore.

How to set the key (only needed for "calorieninjas"):
    Windows (cmd):        set CALORIE_API_KEY=your_key_here
    Windows (PowerShell):  $env:CALORIE_API_KEY="your_key_here"
    macOS / Linux:         export CALORIE_API_KEY=your_key_here

    Or paste it directly into CALORIE_API_KEY below (don't commit a real
    key to a public GitHub repo if you do this).

If nothing is configured, lookup_calories() falls straight through to
the "not found — add it yourself" path. Nothing crashes either way.
"""

import os

# ---------------------------------------------------------------------
# Which provider to use for the online fallback
# ---------------------------------------------------------------------
API_PROVIDER = os.environ.get("NUTRITION_API_PROVIDER", "openfoodfacts")

# ---------------------------------------------------------------------
# CalorieNinjas / API Ninjas settings (only used if API_PROVIDER is set
# to "calorieninjas")
# ---------------------------------------------------------------------
CALORIE_API_KEY = ""  # Option B: paste your key here directly (optional)
CALORIE_API_KEY = os.environ.get("CALORIE_API_KEY", CALORIE_API_KEY)
CALORIE_API_URL = "https://api.calorieninjas.com/v1/nutrition"

# ---------------------------------------------------------------------
# Open Food Facts settings — no key needed, nothing to configure
# ---------------------------------------------------------------------
OFF_API_URL = "https://world.openfoodfacts.org/cgi/search.pl"

# How long to wait for the API before giving up (seconds)
API_TIMEOUT_SECONDS = 5


def has_api_key() -> bool:
    """
    Return True if the online lookup flag is configured for the selected CA1 provider.

    Offline-mode network blocking is enforced cleanly at request time in `_fetch_from_api`.
    """
    if os.environ.get("NUTRITRACK_ONLINE_LOOKUP", "").strip() != "1":
        return False
    if API_PROVIDER == "openfoodfacts":
        return True
    return bool(CALORIE_API_KEY.strip())
