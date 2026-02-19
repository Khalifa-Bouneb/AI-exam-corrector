"""
hybrid_grader.py — Hybrid Grading Pipeline.

Combines:
  • BERT-based cosine similarity (semantic matching)
  • LLM-based scoring & feedback (reasoning + constructive commentary)

Final score formula:
    final_score = α × BERT_normalised_score  +  (1 − α) × LLM_score

Where α ∈ [0, 1] is a tunable weight set in config.ALPHA.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config
import bert_scorer
import llm_feedback
from bert_scorer import BERTScoreResult
from llm_feedback import LLMGradingResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HybridGradingResult:
    """Full result of the hybrid grading pipeline for one answer."""
    question_id: str
    reference_answer: str
    student_answer: str

    # Components
    bert_result: BERTScoreResult
    llm_result: LLMGradingResult

    # Hybrid
    alpha: float
    final_score: float
    max_score: float

    elapsed_seconds: float = 0.0

    @property
    def bert_component(self) -> float:
        """α × BERT score."""
        return round(self.alpha * self.bert_result.normalised_score, 2)

    @property
    def llm_component(self) -> float:
        """(1 − α) × LLM score."""
        return round((1 - self.alpha) * self.llm_result.score, 2)

    @property
    def percentage(self) -> float:
        return round((self.final_score / self.max_score) * 100, 1) if self.max_score else 0

    @property
    def grade_letter(self) -> str:
        pct = self.percentage
        if pct >= 90: return "A+"
        if pct >= 80: return "A"
        if pct >= 70: return "B"
        if pct >= 60: return "C"
        if pct >= 50: return "D"
        return "F"

    @property
    def feedback(self) -> str:
        return self.llm_result.feedback

    def summary(self) -> str:
        return (
            f"Q{self.question_id}  |  Final: {self.final_score:.1f}/{self.max_score} "
            f"({self.percentage}%, {self.grade_letter})  |  "
            f"BERT: {self.bert_result.normalised_score:.1f} × α={self.alpha}  +  "
            f"LLM: {self.llm_result.score:.1f} × (1−α)={1 - self.alpha:.1f}  |  "
            f"Cosine: {self.bert_result.cosine_similarity:.4f}  |  "
            f"Time: {self.elapsed_seconds:.1f}s"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Single answer grading
# ═══════════════════════════════════════════════════════════════════════════════

def grade_answer(
    reference: str,
    student: str,
    question_id: str = "1",
    max_score: Optional[float] = None,
    alpha: Optional[float] = None,
) -> HybridGradingResult:
    """
    Grade a single student answer using the hybrid BERT + LLM pipeline.

    Parameters
    ----------
    reference   : Professor's model answer.
    student     : Student's submitted answer.
    question_id : Identifier for the question (for tracking).
    max_score   : Maximum possible score (default from config).
    alpha       : Blending weight (default from config).

    Returns
    -------
    HybridGradingResult with all components and the final blended score.
    """
    if max_score is None:
        max_score = config.MAX_SCORE
    if alpha is None:
        alpha = config.ALPHA

    t0 = time.time()

    # ── Step 1: BERT Similarity ──────────────────────────────────────────────
    logger.info("[Q%s] Step 1 — Computing BERT embeddings & similarity …", question_id)
    bert_res = bert_scorer.score(reference, student, max_score=max_score)
    logger.info("[Q%s]   BERT cosine=%.4f  score=%.1f/%s",
                question_id, bert_res.cosine_similarity, bert_res.normalised_score, max_score)

    # ── Step 2: LLM Grading ─────────────────────────────────────────────────
    logger.info("[Q%s] Step 2 — Calling LLM for grading & feedback …", question_id)
    llm_res = llm_feedback.grade(
        reference, student,
        cosine_similarity=bert_res.cosine_similarity,
        max_score=max_score,
    )
    logger.info("[Q%s]   LLM  score=%.1f/%s", question_id, llm_res.score, max_score)

    # ── Step 3: Hybrid Score ─────────────────────────────────────────────────
    final = round(alpha * bert_res.normalised_score + (1 - alpha) * llm_res.score, 2)
    # Clamp to [0, max_score]
    final = max(0.0, min(float(max_score), final))

    elapsed = time.time() - t0

    result = HybridGradingResult(
        question_id=question_id,
        reference_answer=reference,
        student_answer=student,
        bert_result=bert_res,
        llm_result=llm_res,
        alpha=alpha,
        final_score=final,
        max_score=max_score,
        elapsed_seconds=round(elapsed, 2),
    )

    logger.info("[Q%s]   FINAL = %.1f/%s  (%s)  [%.1fs]",
                question_id, final, max_score, result.grade_letter, elapsed)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch grading
# ═══════════════════════════════════════════════════════════════════════════════

def grade_exam(
    answer_pairs: List[Tuple[str, str, str]],
    max_score: Optional[float] = None,
    alpha: Optional[float] = None,
) -> List[HybridGradingResult]:
    """
    Grade a full exam (multiple question-answer pairs).

    Parameters
    ----------
    answer_pairs : List of (question_id, reference_answer, student_answer).
    max_score    : Maximum score per question.
    alpha        : Blending weight.

    Returns
    -------
    List of HybridGradingResult, one per question.
    """
    results = []
    for qid, ref, stu in answer_pairs:
        r = grade_answer(ref, stu, question_id=qid, max_score=max_score, alpha=alpha)
        results.append(r)
    return results


def exam_summary(results: List[HybridGradingResult]) -> dict:
    """Compute aggregate statistics for a graded exam."""
    if not results:
        return {}

    total_score = sum(r.final_score for r in results)
    total_max = sum(r.max_score for r in results)
    avg_cosine = sum(r.bert_result.cosine_similarity for r in results) / len(results)
    total_time = sum(r.elapsed_seconds for r in results)

    return {
        "num_questions": len(results),
        "total_score": round(total_score, 1),
        "total_max": total_max,
        "percentage": round((total_score / total_max) * 100, 1) if total_max else 0,
        "average_cosine_similarity": round(avg_cosine, 4),
        "total_time_seconds": round(total_time, 1),
        "per_question": [r.summary() for r in results],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")

    pairs = [
        ("1",
         "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.",
         "Plants use light from the sun to make food from CO2 and water."),
        ("2",
         "The mitochondria is the organelle responsible for producing ATP through cellular respiration.",
         "Mitochondria makes energy for the cell."),
        ("3",
         "Newton's second law states that force equals mass times acceleration (F = ma).",
         "The law says that heavier objects need more force."),
    ]

    results = grade_exam(pairs)
    summary = exam_summary(results)

    print("\n" + "=" * 70)
    print("EXAM SUMMARY")
    print("=" * 70)
    print(f"Total: {summary['total_score']}/{summary['total_max']} ({summary['percentage']}%)")
    print(f"Avg Cosine: {summary['average_cosine_similarity']}")
    print(f"Time: {summary['total_time_seconds']}s")
    print()
    for line in summary["per_question"]:
        print(f"  {line}")
