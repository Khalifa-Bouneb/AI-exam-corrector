"""
bert_scorer.py — BERT-based Semantic Similarity Scoring.

Uses a sentence-transformers model (a BERT variant fine-tuned on semantic
textual similarity tasks) to:
  1. Encode the reference answer and student answer into dense embeddings.
  2. Compute cosine similarity ∈ [0, 1].
  3. Normalise the similarity to a 0–max_score scale.

The model is loaded lazily and cached for the lifetime of the process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

import config

logger = logging.getLogger(__name__)

# ── Lazy-loaded model ────────────────────────────────────────────────────────
_model = None


def _get_model():
    """Load the sentence-transformers model (once)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading BERT model '%s' …", config.BERT_MODEL_NAME)
        _model = SentenceTransformer(config.BERT_MODEL_NAME)
        logger.info(
            "Model loaded  —  dim=%d, max_seq=%d",
            _model.get_sentence_embedding_dimension(),
            _model.max_seq_length,
        )
    return _model


# ═══════════════════════════════════════════════════════════════════════════════
#  Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def encode(texts: str | List[str], batch_size: int = 32) -> NDArray[np.float32]:
    """
    Encode one or more texts into L2-normalised embeddings.

    Parameters
    ----------
    texts : a single string or a list of strings.
    batch_size : inference batch size.

    Returns
    -------
    numpy array of shape (n, dim) with unit-norm rows.
    """
    model = _get_model()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # L2-normalise → cosine = dot product
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity(vec_a: NDArray, vec_b: NDArray) -> float:
    """
    Cosine similarity between two vectors (or batch means).
    If inputs are already L2-normalised the dot product suffices.
    """
    a = vec_a.flatten()
    b = vec_b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data class for results
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BERTScoreResult:
    """Result container for a single BERT grading."""
    cosine_similarity: float          # raw value ∈ [-1, 1], practically [0, 1]
    normalised_score: float           # mapped to [0, max_score]
    max_score: float
    model_name: str
    embedding_dim: int

    @property
    def percentage(self) -> float:
        return round(self.cosine_similarity * 100, 2)

    def __repr__(self) -> str:
        return (
            f"BERTScore(cosine={self.cosine_similarity:.4f}, "
            f"score={self.normalised_score:.2f}/{self.max_score}, "
            f"model='{self.model_name}')"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Public grading function
# ═══════════════════════════════════════════════════════════════════════════════

def score(
    reference: str,
    student: str,
    max_score: Optional[float] = None,
) -> BERTScoreResult:
    """
    Score a student answer against a reference answer using BERT embeddings.

    Parameters
    ----------
    reference : The professor's model / reference answer.
    student   : The student's submitted answer.
    max_score : Maximum possible score (defaults to config.MAX_SCORE).

    Returns
    -------
    BERTScoreResult with cosine similarity and normalised score.
    """
    if max_score is None:
        max_score = config.MAX_SCORE

    emb = encode([reference, student])
    cos_sim = cosine_similarity(emb[0], emb[1])

    # Clamp cosine to [0, 1] for scoring (negative = completely unrelated)
    clamped = max(0.0, min(1.0, cos_sim))

    normalised = round(clamped * max_score, 2)

    model = _get_model()
    return BERTScoreResult(
        cosine_similarity=round(cos_sim, 6),
        normalised_score=normalised,
        max_score=max_score,
        model_name=config.BERT_MODEL_NAME,
        embedding_dim=model.get_sentence_embedding_dimension(),
    )


def score_batch(
    references: List[str],
    students: List[str],
    max_score: Optional[float] = None,
) -> List[BERTScoreResult]:
    """
    Score multiple (reference, student) answer pairs in a single batch.
    More efficient than calling score() in a loop.
    """
    if max_score is None:
        max_score = config.MAX_SCORE

    assert len(references) == len(students), "Mismatched list lengths"

    ref_embs = encode(references)
    stu_embs = encode(students)

    model = _get_model()
    dim = model.get_sentence_embedding_dimension()

    results: List[BERTScoreResult] = []
    for ref_e, stu_e in zip(ref_embs, stu_embs):
        cos_sim = cosine_similarity(ref_e, stu_e)
        clamped = max(0.0, min(1.0, cos_sim))
        results.append(BERTScoreResult(
            cosine_similarity=round(cos_sim, 6),
            normalised_score=round(clamped * max_score, 2),
            max_score=max_score,
            model_name=config.BERT_MODEL_NAME,
            embedding_dim=dim,
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    ref = "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water."
    stu = "Plants use light from the sun to make food from CO2 and water."

    result = score(ref, stu)
    print(f"\nReference : {ref}")
    print(f"Student   : {stu}")
    print(f"\n{result}")
    print(f"Similarity: {result.percentage}%")
