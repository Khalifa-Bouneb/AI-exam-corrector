"""
test_pipeline.py — End-to-end test of the hybrid grading pipeline.

Runs a few built-in sample answer pairs through BERT + LLM grading
and prints detailed results.  No dataset download required.

Usage:
    python test_pipeline.py
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Test Data ────────────────────────────────────────────────────────────────

SAMPLES = [
    {
        "question_id": "1",
        "question": "What is photosynthesis?",
        "reference": (
            "Photosynthesis is the process by which green plants and some other "
            "organisms use sunlight to synthesize nutrients from carbon dioxide "
            "and water. It generally involves the green pigment chlorophyll and "
            "generates oxygen as a by-product."
        ),
        "student_answer": (
            "Plants use light from the sun to make food from CO2 and water."
        ),
        "expected_quality": "Good — captures essence but misses chlorophyll & O2",
    },
    {
        "question_id": "2",
        "question": "What is the role of mitochondria?",
        "reference": (
            "The mitochondria is the organelle responsible for producing ATP "
            "through the process of cellular respiration. It is often called "
            "the powerhouse of the cell."
        ),
        "student_answer": (
            "Mitochondria makes energy for the cell."
        ),
        "expected_quality": "Average — vague, missing ATP & cellular respiration",
    },
    {
        "question_id": "3",
        "question": "State Newton's second law.",
        "reference": (
            "Newton's second law states that the acceleration of an object "
            "is directly proportional to the net force acting on it and "
            "inversely proportional to its mass. Mathematically, F = ma."
        ),
        "student_answer": (
            "The law says that heavier objects need more force to move."
        ),
        "expected_quality": "Poor — incomplete, missing F=ma and proportionality",
    },
    {
        "question_id": "4",
        "question": "Explain osmosis.",
        "reference": (
            "Osmosis is the movement of water molecules through a semipermeable "
            "membrane from a region of lower solute concentration to a region of "
            "higher solute concentration, until equilibrium is reached."
        ),
        "student_answer": (
            "Osmosis is the diffusion of water across a selectively permeable "
            "membrane from an area of low solute concentration to high solute "
            "concentration."
        ),
        "expected_quality": "Excellent — nearly identical meaning",
    },
    {
        "question_id": "5",
        "question": "What is DNA replication?",
        "reference": (
            "DNA replication is a semiconservative process where the double helix "
            "unwinds and each strand serves as a template for a new complementary "
            "strand, resulting in two identical DNA molecules."
        ),
        "student_answer": (
            "I don't know."
        ),
        "expected_quality": "Fail — no attempt",
    },
]


def main():
    import config
    print("\n" + "=" * 70)
    print("  AI EXAM CORRECTING — PIPELINE TEST")
    print("=" * 70)
    print(f"  BERT model  : {config.BERT_MODEL_NAME}")
    print(f"  LLM provider: {config.LLM_PROVIDER}")
    print(f"  LLM model   : {config.LLM_MODELS.get(config.LLM_PROVIDER)}")
    print(f"  α (BERT wt) : {config.ALPHA}")
    print(f"  Max score   : {config.MAX_SCORE}")
    print("=" * 70)

    # ── Step 1: Test BERT scorer alone ───────────────────────────────────────
    print("\n▸ Step 1: Testing BERT Scorer …")
    import bert_scorer

    for s in SAMPLES:
        result = bert_scorer.score(s["reference"], s["student_answer"], max_score=config.MAX_SCORE)
        print(f"  Q{s['question_id']}: cosine={result.cosine_similarity:.4f}  "
              f"bert_score={result.normalised_score:.1f}/{config.MAX_SCORE}  "
              f"({result.percentage}%)")

    # ── Step 2: Test full hybrid pipeline ────────────────────────────────────
    print("\n▸ Step 2: Testing Hybrid Pipeline (BERT + LLM) …")
    import hybrid_grader

    pairs = [(s["question_id"], s["reference"], s["student_answer"]) for s in SAMPLES]
    t0 = time.time()
    results = hybrid_grader.grade_exam(pairs, max_score=config.MAX_SCORE)
    elapsed = time.time() - t0

    print("\n" + "─" * 70)
    print(f"{'Q#':<4} {'Final':>8} {'Grade':>6} {'BERT':>8} {'LLM':>8} {'Cosine':>8}  Feedback")
    print("─" * 70)

    for r, s in zip(results, SAMPLES):
        print(
            f"Q{r.question_id:<3} "
            f"{r.final_score:>5.1f}/{r.max_score:<2.0f} "
            f"{r.grade_letter:>5} "
            f"{r.bert_result.normalised_score:>6.1f} "
            f"{r.llm_result.score:>6.1f} "
            f"{r.bert_result.cosine_similarity:>7.4f}  "
            f"{r.feedback[:60]}…"
        )

    print("─" * 70)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = hybrid_grader.exam_summary(results)
    print(f"\n  Total: {summary['total_score']}/{summary['total_max']} ({summary['percentage']}%)")
    print(f"  Avg Cosine: {summary['average_cosine_similarity']}")
    print(f"  Pipeline time: {elapsed:.1f}s  ({elapsed/len(SAMPLES):.1f}s per question)")
    print(f"  Formula: final = {config.ALPHA} × BERT + {1-config.ALPHA} × LLM")
    print("\n✅ Pipeline test complete.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Pipeline test failed: %s", e, exc_info=True)
        sys.exit(1)
