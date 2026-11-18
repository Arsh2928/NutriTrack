"""Test package.

Redirects ALL runtime storage to a throw-away temp folder BEFORE any
nutritrack module is imported (food_database reads custom_foods.json at import
time). Tests therefore never read or modify real data in ./data or ./logs,
never use the network, and never start the LLM agent.
"""
import atexit
import os
import shutil
import tempfile

_ROOT = tempfile.mkdtemp(prefix="nutritrack_tests_")
os.environ["NUTRITRACK_DATA_DIR"] = os.path.join(_ROOT, "data")
os.environ["NUTRITRACK_LOG_DIR"] = os.path.join(_ROOT, "logs")
os.environ["NUTRITRACK_CUSTOM_FOODS"] = os.path.join(_ROOT, "data", "custom_foods.json")
os.environ["NUTRITION_API_PROVIDER"] = "calorieninjas"   # + no key => online food lookup OFF
os.environ.pop("CALORIE_API_KEY", None)
os.environ.pop("GITHUB_TOKEN", None)                      # force the offline rule-based agent
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)
