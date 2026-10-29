"""Central settings. Everything can be overridden with environment variables or a .env file."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
CASES_DIR = ROOT / "cases"                 # the six frozen demo cases
CASES_DEV_DIR = ROOT / "cases_dev"         # development cases used while tuning (never used in the final comparison)
CHECKLISTS_DIR = ROOT / "checklists"        # expected behaviour; never read by the app or the pipeline
CARDS_PATH = DATA_DIR / "cards.jsonl"

# One fixed generation model for BOTH modes (record the exact name in the README).
MODEL_NAME = os.getenv("AIMSCRIBE_MODEL", "gemini-3.5-flash-lite")
GEN_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = int(os.getenv("AIMSCRIBE_MAX_TOKENS", "4096"))

# Retrieval
EMBED_BACKEND = os.getenv("AIMSCRIBE_EMBED", "fastembed")     # "fastembed" or "tfidf"
EMBED_MODEL = os.getenv("AIMSCRIBE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = int(os.getenv("AIMSCRIBE_TOPK", "8"))
RETRIEVAL = os.getenv("AIMSCRIBE_RETRIEVAL", "hybrid")      # dense | bm25 | hybrid (reciprocal rank fusion)
ROUTE_DISEASES = int(os.getenv("AIMSCRIBE_ROUTE", "0"))      # 0 = off; N = keep cards only from up to N best-scoring diseases
ROUTE_RATIO = float(os.getenv("AIMSCRIBE_ROUTE_RATIO", "0"))  # a 2nd..Nth disease is kept only if its score >= ratio x best score (0 = always keep N)

# Free tier of gemini-3.5-flash-lite allows 15 requests/minute; stay well under it.
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("AIMSCRIBE_MIN_GAP", "4.5"))
MAX_RETRIES = 4
