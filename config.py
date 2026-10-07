"""Global configuration and constants."""
import os
from pathlib import Path

# ============ Paths ============
PROJECT_DIR = Path(__file__).resolve().parent
FEABENCH_DATA_PATH = Path(os.environ.get("FEABENCH_DATA_PATH", str(
    PROJECT_DIR / "feabench-data/FEA-Bench-v1.0-Lite-Standard/test/data-00000-of-00001.arrow")))
REPOS_DIR = Path(os.environ.get("FEABENCH_REPOS_DIR", str(PROJECT_DIR / "repos")))
OUTPUT_DIR = Path(os.environ.get("FUNCGEN_OUTPUT_DIR", str(PROJECT_DIR / "outputs")))

# CoderEval
CODEREVAL_DATA = Path("/data2/ymz/codereval_data/data/converted.jsonl")
CODEREVAL_REPOS = Path("/data2/ymz/codereval_data/CoderEval-Docker/home/travis/builds/repos")

# ============ LLM Config ============
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
LLM_MODEL_UNDERSTAND = os.environ.get("LLM_MODEL_UNDERSTAND", "deepseek-chat")
LLM_MODEL_PATTERN = os.environ.get("LLM_MODEL_PATTERN", "deepseek-chat")
LLM_MODEL_GENERATE = os.environ.get("LLM_MODEL_GENERATE", "deepseek-chat")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "8192"))
LLM_PLAN_MAX_TOKENS = int(os.environ.get("LLM_PLAN_MAX_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")

# ============ Explorer Config ============
MAX_IMPORT_DEPTH = 5          # Max depth for import chain BFS
MAX_CONTEXT_CHARS = 80000     # Max total context sent to LLM
MAX_FILE_CHARS = 20000        # Max chars per file read
MAX_SIBLING_METHODS = 15      # Max sibling methods to include in context
SIMILARITY_THRESHOLD = 0.3    # Min signature overlap to consider "similar"

# ============ Repo Config ============
GIT_CHECKOUT_TIMEOUT = 30     # Seconds
CONDA_ENV = "py38"

# ============ Validation ============
TEST_TIMEOUT = 120            # Seconds per test run
