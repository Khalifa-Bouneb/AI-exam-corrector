"""
llm_feedback.py — LLM-based grading and feedback generation.

Sends the reference answer, student answer, and BERT cosine similarity
to a free LLM API (Groq / Gemini / OpenRouter).

The LLM returns:
  • A numeric score (0 – max_score)
  • Constructive textual feedback

Supports: Groq, Google Gemini, OpenRouter.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Result data-class
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMGradingResult:
    """Container for the LLM grading output."""
    score: float                   # 0 – max_score
    feedback: str                  # constructive text
    max_score: float
    provider: str
    model: str
    raw_response: str = ""         # raw LLM text for debugging

    @property
    def normalised(self) -> float:
        """Score normalised to [0, 1]."""
        return self.score / self.max_score if self.max_score else 0.0

    def __repr__(self) -> str:
        return (
            f"LLMGrade(score={self.score}/{self.max_score}, "
            f"provider='{self.provider}', model='{self.model}')"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  System prompt construction
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(max_score: float) -> str:
    return f"""You are an expert, fair, and constructive academic grader for a short-answer exam.

CONTEXT:
- You receive a "Reference Answer" (professor's ideal), a "Student Answer", and a
  pre-computed "Cosine Similarity" (0-1) from a BERT-based embedding model.
- Maximum possible score: {max_score}.

INSTRUCTIONS:
1. Use the Cosine Similarity as a SEMANTIC BASELINE showing how close the meanings are.
2. Apply expert judgement ON TOP of the baseline:
   a. Reward correct key concepts even if phrased differently.
   b. Penalize factual errors, missing critical points, or irrelevant content.
   c. Award partial credit when partly correct.
3. The score MUST be a number between 0 and {max_score} (can be a float with 1 decimal).
4. Write 1-3 sentences of constructive feedback — mention strengths AND weaknesses.

OUTPUT:
Return ONLY a valid JSON object (no markdown fences, no extra text):
{{"score": <number>, "feedback": "<text>"}}"""


def _build_user_message(reference: str, student: str, cosine_sim: float) -> str:
    return f"""Reference Answer:
\"\"\"
{reference}
\"\"\"

Student Answer:
\"\"\"
{student}
\"\"\"

Cosine Similarity: {cosine_sim:.4f}  (0 = unrelated, 1 = identical meaning)

Grade the student's answer now. Respond ONLY with the JSON object."""


# ═══════════════════════════════════════════════════════════════════════════════
#  API callers
# ═══════════════════════════════════════════════════════════════════════════════

def _call_groq(api_key: str, model: str, system: str, user: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": 400,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(api_key: str, model: str, system: str, user: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": config.LLM_TEMPERATURE,
            "maxOutputTokens": 400,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_openrouter(api_key: str, model: str, system: str, user: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Title": "AI-Correcting-ASAG",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": 400,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


_CALLERS = {
    "groq":       _call_groq,
    "gemini":     _call_gemini,
    "openrouter": _call_openrouter,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Response parser
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_response(raw: str, max_score: float) -> dict:
    """Parse LLM raw text → {"score": float, "feedback": str}."""
    cleaned = raw.strip()
    # Strip markdown fences
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    try:
        obj = json.loads(cleaned)
        s = max(0, min(max_score, round(float(obj["score"]), 1)))
        return {"score": s, "feedback": str(obj.get("feedback", ""))}
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Regex fallback
    score_m = re.search(r'"score"\s*:\s*([\d.]+)', cleaned)
    fb_m = re.search(r'"feedback"\s*:\s*"([^"]+)"', cleaned)
    if score_m:
        s = max(0, min(max_score, round(float(score_m.group(1)), 1)))
        fb = fb_m.group(1) if fb_m else "Could not parse feedback."
        return {"score": s, "feedback": fb}

    raise ValueError(f"Cannot parse LLM response: {raw[:300]}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def grade(
    reference: str,
    student: str,
    cosine_similarity: float,
    max_score: Optional[float] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMGradingResult:
    """
    Ask an LLM to grade a student answer.

    Parameters
    ----------
    reference        : Professor's reference answer.
    student          : Student's submitted answer.
    cosine_similarity: Pre-computed BERT cosine similarity [0, 1].
    max_score        : Maximum score (default from config).
    provider         : LLM provider override (default from config).
    model            : Model override (default from config).
    api_key          : API key override (default from config).

    Returns
    -------
    LLMGradingResult with score, feedback, and metadata.
    """
    if max_score is None:
        max_score = config.MAX_SCORE
    if provider is None:
        provider = config.LLM_PROVIDER
    if model is None:
        model = config.LLM_MODELS.get(provider, "")
    if api_key is None:
        api_key = config.get_api_key()

    caller = _CALLERS.get(provider)
    if caller is None:
        raise ValueError(f"Unknown LLM provider: {provider}")

    system_prompt = _build_system_prompt(max_score)
    user_message = _build_user_message(reference, student, cosine_similarity)

    logger.info("Calling LLM  provider=%s  model=%s", provider, model)
    raw = caller(api_key, model, system_prompt, user_message)
    logger.debug("LLM raw response: %s", raw[:500])

    parsed = _parse_response(raw, max_score)

    return LLMGradingResult(
        score=parsed["score"],
        feedback=parsed["feedback"],
        max_score=max_score,
        provider=provider,
        model=model,
        raw_response=raw,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    ref = "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water."
    stu = "Plants use light from the sun to make food from CO2 and water."

    result = grade(ref, stu, cosine_similarity=0.82)
    print(f"\n{result}")
    print(f"Feedback: {result.feedback}")
