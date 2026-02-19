"""
dataset_loader.py — Load, prepare, and evaluate on standard ASAG datasets.

Supported datasets (all free / publicly available):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Mohler et al. (2011) — "Learning to Grade Short Answer Questions using
   Semantic Similarity Measures and Dependency Graph Alignments"
   • 2,273 student answers to 80 questions from an intro CS course.
   • Scores: 0-5 (continuous, 2 human annotators).
   • Download: https://github.com/lm-pub-quiz/Mohler-dataset
     or https://www.kaggle.com/datasets/smiles28/short-answer-grading

2. SemEval-2013 Task 7 — Student Response Analysis
   • Beetle + SciEntsBank subsets.
   • 5-way labels: correct / partially_correct_incomplete / contradictory /
     irrelevant / non_domain.
   • Download: https://www.cs.york.ac.uk/semeval-2013/task7/
     or via 🤗 `datasets` library: "semeval2013_task7"

3. ASAP-SAS (Kaggle) — Automated Student Assessment Prize – Short Answer Scoring
   • ~17k answers across 10 question prompts.
   • Scores: 0-2 or 0-3 per prompt.
   • Download: https://www.kaggle.com/c/asap-sas/data

4. Texas Dataset (Mohler & Mihalcea, 2009)
   • Subset predecessor to #1.  Often bundled together.

5. SciEntsBank (Dzikovska et al., 2012)
   • Incorporated into SemEval-2013, but also standalone releases.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module provides loaders that return a unified format:
    List[dict] with keys: question_id, reference, student_answer, gold_score, max_score

And an evaluate() function that runs our hybrid pipeline on a dataset and
computes Pearson / Spearman correlation, RMSE, and QWK against gold scores.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Unified row format
ROW_KEYS = ("question_id", "reference", "student_answer", "gold_score", "max_score")


# ═══════════════════════════════════════════════════════════════════════════════
#  Mohler Dataset Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_mohler(data_dir: Optional[str] = None) -> List[dict]:
    """
    Load the Mohler et al. (2011) short-answer grading dataset.

    Expected structure inside `data_dir`:
        data_dir/
        ├── raw/
        │   ├── questions/         ← one .txt per question
        │   ├── answers/           ← one .txt per question, lines = student answers
        │   └── scores/            ← one .txt per question, lines = gold scores
        OR
        ├── mohler.csv             ← single CSV with columns:
        │                            question_id, question, ref_answer,
        │                            student_answer, score1, score2

    Returns list of dicts with unified keys.
    """
    if data_dir is None:
        data_dir = config.DATASETS_DIR / "mohler"
    data_dir = Path(data_dir)

    # Try CSV first (easier format)
    csv_path = data_dir / "mohler.csv"
    if csv_path.exists():
        return _load_mohler_csv(csv_path)

    # Try raw folder structure
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        return _load_mohler_raw(raw_dir)

    raise FileNotFoundError(
        f"Mohler dataset not found at {data_dir}. "
        f"Please download from: https://github.com/lm-pub-quiz/Mohler-dataset "
        f"and place files in {data_dir}."
    )


def _load_mohler_csv(csv_path: Path) -> List[dict]:
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        # Average the two annotator scores
        s1 = float(r.get("score1", r.get("score_me", r.get("score", 0))))
        s2 = float(r.get("score2", r.get("score_other", s1)))
        gold = (s1 + s2) / 2.0
        rows.append({
            "question_id": str(r.get("question_id", r.get("id", ""))),
            "reference": str(r.get("ref_answer", r.get("reference", ""))),
            "student_answer": str(r.get("student_answer", r.get("answer", ""))),
            "gold_score": round(gold, 2),
            "max_score": 5.0,
        })
    logger.info("Loaded Mohler CSV: %d samples", len(rows))
    return rows


def _load_mohler_raw(raw_dir: Path) -> List[dict]:
    questions_dir = raw_dir / "questions"
    answers_dir = raw_dir / "answers"
    scores_dir = raw_dir / "scores"
    rows = []

    for q_file in sorted(questions_dir.glob("*.txt")):
        qid = q_file.stem
        ref_text = q_file.read_text(encoding="utf-8").strip()
        ans_file = answers_dir / f"{qid}.txt"
        scr_file = scores_dir / f"{qid}.txt"
        if not ans_file.exists() or not scr_file.exists():
            continue

        answers = ans_file.read_text(encoding="utf-8").strip().splitlines()
        scores = scr_file.read_text(encoding="utf-8").strip().splitlines()

        for ans, sc in zip(answers, scores):
            parts = sc.split()
            s_vals = [float(x) for x in parts if x.replace(".", "").replace("-", "").isdigit()]
            gold = sum(s_vals) / len(s_vals) if s_vals else 0
            rows.append({
                "question_id": qid,
                "reference": ref_text,
                "student_answer": ans.strip(),
                "gold_score": round(gold, 2),
                "max_score": 5.0,
            })

    logger.info("Loaded Mohler raw: %d samples", len(rows))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  SemEval-2013 Task 7 Loader (via HuggingFace `datasets`)
# ═══════════════════════════════════════════════════════════════════════════════

SEMEVAL_LABEL_MAP = {
    "correct": 1.0,
    "partially_correct_incomplete": 0.5,
    "contradictory": 0.1,
    "irrelevant": 0.0,
    "non_domain": 0.0,
}

def load_semeval(subset: str = "beetle", split: str = "test") -> List[dict]:
    """
    Load SemEval-2013 Task 7 via HuggingFace `datasets` library.

    Parameters
    ----------
    subset : "beetle" or "sciEntsBank"
    split  : "train", "test", "unseen_answers", "unseen_questions", "unseen_domains"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets` library: pip install datasets")

    ds = load_dataset("semeval2013_task7", subset, split=split, trust_remote_code=True)
    rows = []
    for item in ds:
        label = item.get("label", item.get("accuracy", ""))
        if isinstance(label, int):
            label_names = ["correct", "partially_correct_incomplete",
                           "contradictory", "irrelevant", "non_domain"]
            label = label_names[label] if label < len(label_names) else "irrelevant"
        gold = SEMEVAL_LABEL_MAP.get(label, 0.0)
        rows.append({
            "question_id": str(item.get("id", "")),
            "reference": str(item.get("reference_answer", "")),
            "student_answer": str(item.get("student_answer", "")),
            "gold_score": gold,
            "max_score": 1.0,
        })
    logger.info("Loaded SemEval '%s/%s': %d samples", subset, split, len(rows))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  ASAP-SAS (Kaggle) Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_asap_sas(data_dir: Optional[str] = None) -> List[dict]:
    """
    Load the ASAP-SAS dataset from Kaggle (requires manual download).

    Expected: data_dir/train.tsv with columns:
      Id, EssaySet, Score1, Score2, EssayText
    """
    if data_dir is None:
        data_dir = config.DATASETS_DIR / "asap-sas"
    data_dir = Path(data_dir)

    tsv_path = data_dir / "train.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"ASAP-SAS not found at {tsv_path}. "
            f"Download from: https://www.kaggle.com/c/asap-sas/data"
        )

    df = pd.read_csv(tsv_path, sep="\t")
    # ASAP-SAS doesn't have explicit reference answers — they're in the rubric
    # We'll use a prompt-level reference if a rubric file exists
    rubric = _load_asap_rubrics(data_dir)

    rows = []
    for _, r in df.iterrows():
        prompt_id = str(r.get("EssaySet", ""))
        s1 = float(r.get("Score1", 0))
        s2 = float(r.get("Score2", s1))
        gold = (s1 + s2) / 2.0
        max_s = {1: 3, 2: 3, 3: 2, 4: 2, 5: 3, 6: 3, 7: 2, 8: 2, 9: 2, 10: 2}.get(int(prompt_id), 3)

        rows.append({
            "question_id": prompt_id,
            "reference": rubric.get(prompt_id, "No reference available for this prompt."),
            "student_answer": str(r.get("EssayText", "")),
            "gold_score": gold,
            "max_score": float(max_s),
        })

    logger.info("Loaded ASAP-SAS: %d samples", len(rows))
    return rows


def _load_asap_rubrics(data_dir: Path) -> Dict[str, str]:
    """Try to load reference answers / rubrics for ASAP-SAS prompts."""
    rubric_path = data_dir / "rubrics.json"
    if rubric_path.exists():
        with open(rubric_path) as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  Generic CSV / JSON loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_generic_csv(
    path: str | Path,
    ref_col: str = "reference",
    stu_col: str = "student_answer",
    score_col: str = "score",
    max_score: float = 5.0,
    qid_col: Optional[str] = None,
) -> List[dict]:
    """Load any CSV with reference, student answer, and score columns."""
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        rows.append({
            "question_id": str(r[qid_col]) if qid_col and qid_col in df.columns else str(i),
            "reference": str(r[ref_col]),
            "student_answer": str(r[stu_col]),
            "gold_score": float(r[score_col]),
            "max_score": max_score,
        })
    logger.info("Loaded generic CSV: %d samples from %s", len(rows), path)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    gold_scores: List[float],
    predicted_scores: List[float],
    max_score: float = 5.0,
) -> Dict[str, float]:
    """
    Compute evaluation metrics: Pearson r, Spearman ρ, RMSE, MAE, QWK.

    Parameters
    ----------
    gold_scores      : ground-truth scores
    predicted_scores : model-predicted scores
    max_score        : for QWK discretisation

    Returns dict with keys: pearson, spearman, rmse, mae, qwk
    """
    from scipy.stats import pearsonr, spearmanr

    gold = np.array(gold_scores)
    pred = np.array(predicted_scores)

    pearson_r, _ = pearsonr(gold, pred)
    spearman_r, _ = spearmanr(gold, pred)
    rmse = float(np.sqrt(np.mean((gold - pred) ** 2)))
    mae = float(np.mean(np.abs(gold - pred)))

    # Quadratic Weighted Kappa — discretise to integers
    gold_int = np.round(gold).astype(int)
    pred_int = np.round(pred).astype(int)
    qwk = _quadratic_weighted_kappa(gold_int, pred_int, int(max_score))

    return {
        "pearson": round(pearson_r, 4),
        "spearman": round(spearman_r, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "qwk": round(qwk, 4),
    }


def _quadratic_weighted_kappa(y_true, y_pred, max_score: int) -> float:
    """Compute QWK between two integer arrays."""
    n_classes = max_score + 1
    # Confusion matrix
    O = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        t = min(max(0, t), max_score)
        p = min(max(0, p), max_score)
        O[t][p] += 1

    N = len(y_true)
    if N == 0:
        return 0.0

    # Weight matrix
    W = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            W[i][j] = (i - j) ** 2 / (max_score ** 2)

    # Expected matrix
    row_sum = O.sum(axis=1)
    col_sum = O.sum(axis=0)
    E = np.outer(row_sum, col_sum) / N

    num = np.sum(W * O)
    den = np.sum(W * E)
    return 1.0 - num / den if den != 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Full evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_dataset(
    dataset: List[dict],
    max_samples: Optional[int] = None,
    alpha: Optional[float] = None,
) -> Dict:
    """
    Run the hybrid grading pipeline on a dataset and compute metrics.

    Parameters
    ----------
    dataset     : list of dicts with unified keys (from any loader).
    max_samples : limit for quick tests (None = use all).
    alpha       : override for config.ALPHA.

    Returns
    -------
    dict with keys: metrics, predictions, num_samples
    """
    import hybrid_grader

    if max_samples:
        dataset = dataset[:max_samples]

    gold_scores = []
    pred_scores = []
    details = []

    for i, row in enumerate(dataset):
        logger.info("Evaluating sample %d/%d  (Q%s)", i + 1, len(dataset), row["question_id"])
        try:
            result = hybrid_grader.grade_answer(
                reference=row["reference"],
                student=row["student_answer"],
                question_id=row["question_id"],
                max_score=row["max_score"],
                alpha=alpha,
            )
            # Normalise both to [0, 1] for fair comparison
            gold_norm = row["gold_score"] / row["max_score"]
            pred_norm = result.final_score / result.max_score

            gold_scores.append(row["gold_score"])
            pred_scores.append(result.final_score)

            details.append({
                "question_id": row["question_id"],
                "gold": row["gold_score"],
                "predicted": result.final_score,
                "bert_cosine": result.bert_result.cosine_similarity,
                "llm_score": result.llm_result.score,
                "feedback": result.feedback,
            })
        except Exception as e:
            logger.error("Error on sample %d: %s", i, e)
            continue

    metrics = compute_metrics(gold_scores, pred_scores, max_score=dataset[0]["max_score"])

    return {
        "num_samples": len(gold_scores),
        "metrics": metrics,
        "predictions": details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")

    # Print dataset references
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ASAG DATASET REFERENCES                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. Mohler et al. (2011)                                           ║
║     "Learning to Grade Short Answer Questions using Semantic       ║
║      Similarity Measures and Dependency Graph Alignments"          ║
║     → 2,273 answers, 80 questions, scores 0–5                     ║
║     → https://github.com/lm-pub-quiz/Mohler-dataset               ║
║     → https://kaggle.com/datasets/smiles28/short-answer-grading    ║
║                                                                    ║
║  2. SemEval-2013 Task 7 — Student Response Analysis                ║
║     Beetle + SciEntsBank datasets                                  ║
║     → 5-way classification (correct / partial / contradictory …)   ║
║     → HuggingFace: datasets.load_dataset("semeval2013_task7")      ║
║     → https://www.cs.york.ac.uk/semeval-2013/task7/                ║
║                                                                    ║
║  3. ASAP-SAS (Kaggle Competition)                                  ║
║     → ~17k answers, 10 prompts, scores 0–2 or 0–3                 ║
║     → https://www.kaggle.com/c/asap-sas/data                      ║
║                                                                    ║
║  4. Texas Dataset (Mohler & Mihalcea, 2009)                        ║
║     → Precursor to #1, often bundled together                      ║
║                                                                    ║
║  5. SciEntsBank (Dzikovska et al., 2012)                           ║
║     → Science domain, incorporated in SemEval-2013                 ║
║     → Standalone: https://www.cs.york.ac.uk/                       ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Quick test with built-in sample
    sample_data = [
        {
            "question_id": "1",
            "reference": "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.",
            "student_answer": "Plants use light from the sun to make food from CO2 and water.",
            "gold_score": 4.2,
            "max_score": 5.0,
        },
        {
            "question_id": "2",
            "reference": "The mitochondria is the organelle responsible for producing ATP through cellular respiration.",
            "student_answer": "Mitochondria makes energy for the cell.",
            "gold_score": 3.0,
            "max_score": 5.0,
        },
    ]

    print("Running evaluation on 2 built-in samples…\n")
    result = evaluate_on_dataset(sample_data)
    print(f"\nMetrics: {result['metrics']}")
    for p in result["predictions"]:
        print(f"  Q{p['question_id']}: gold={p['gold']}, pred={p['predicted']:.1f}, cosine={p['bert_cosine']:.4f}")
