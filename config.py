"""
config.py — Central configuration for the AI Correcting pipeline.

All API keys are loaded from environment variables (.env file).
Never hardcode secrets here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ---------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

# ── Paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
SAMPLES_DIR  = PROJECT_ROOT / "samples"

# ── API Keys (loaded from environment / .env) --------------------------------
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── LLM Settings -------------------------------------------------------------
# Options: "groq", "gemini", "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

LLM_MODELS = {
    "groq":       os.getenv("GROQ_MODEL",       "llama-3.3-70b-versatile"),
    "gemini":     os.getenv("GEMINI_MODEL",      "gemini-2.0-flash"),
    "openrouter": os.getenv("OPENROUTER_MODEL",  "meta-llama/llama-3.3-70b-instruct:free"),
}

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ── BERT Embedding Model -----------------------------------------------------
# Any sentence-transformers model name from HuggingFace Hub:
#   - "all-MiniLM-L6-v2"             (fast, 384-dim, English)
#   - "paraphrase-mpnet-base-v2"     (higher quality, 768-dim)
#   - "bert-base-nli-stsb-mean-tokens"
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "all-MiniLM-L6-v2")

# ── Hybrid Scoring -----------------------------------------------------------
# final_score = ALPHA * bert_similarity + (1 - ALPHA) * llm_normalized_score
# Tune ALPHA between 0.0 (pure LLM) and 1.0 (pure BERT).
ALPHA = float(os.getenv("ALPHA", "0.4"))

# ── OCR Settings -------------------------------------------------------------
# Options: "easyocr", "tesseract"
OCR_ENGINE = os.getenv("OCR_ENGINE", "easyocr")
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en,fr").split(",")

# ── Grading Scale ------------------------------------------------------------
MAX_SCORE = int(os.getenv("MAX_SCORE", "20"))

# ── Validation Helper --------------------------------------------------------
def get_api_key() -> str:
    """Return the API key for the currently selected LLM provider."""
    mapping = {
        "groq":       GROQ_API_KEY,
        "gemini":     GEMINI_API_KEY,
        "openrouter": OPENROUTER_API_KEY,
    }
    key = mapping.get(LLM_PROVIDER, "")
    if not key:
        raise ValueError(
            f"No API key found for provider '{LLM_PROVIDER}'. "
            f"Set the corresponding env var in your .env file."
        )
    return key
