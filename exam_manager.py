"""
exam_manager.py — Manages exam templates (reference answers stored as JSON).

The teacher configures the exam ONCE (question + reference answer for each).
When a student uploads their exam sheet, the system just OCRs and grades
against the stored references — no manual input needed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

import config

logger = logging.getLogger(__name__)

EXAMS_DIR = config.PROJECT_ROOT / "exams"
EXAMS_DIR.mkdir(exist_ok=True)


@dataclass
class Question:
    number: int
    question_text: str
    reference_answer: str
    max_score: float = 5.0


@dataclass
class ExamTemplate:
    exam_id: str
    title: str
    subject: str
    created_at: str = ""
    questions: List[Question] = field(default_factory=list)
    alpha: float = 0.4
    total_max_score: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.total_max_score = sum(q.max_score for q in self.questions)


def save_exam(exam: ExamTemplate) -> Path:
    """Save an exam template to JSON."""
    path = EXAMS_DIR / f"{exam.exam_id}.json"
    data = {
        "exam_id": exam.exam_id,
        "title": exam.title,
        "subject": exam.subject,
        "created_at": exam.created_at,
        "alpha": exam.alpha,
        "questions": [
            {
                "number": q.number,
                "question_text": q.question_text,
                "reference_answer": q.reference_answer,
                "max_score": q.max_score,
            }
            for q in exam.questions
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Exam saved: %s (%d questions)", exam.exam_id, len(exam.questions))
    return path


def load_exam(exam_id: str) -> ExamTemplate:
    """Load an exam template from JSON."""
    path = EXAMS_DIR / f"{exam_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Exam '{exam_id}' not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = [Question(**q) for q in data.get("questions", [])]
    return ExamTemplate(
        exam_id=data["exam_id"],
        title=data["title"],
        subject=data["subject"],
        created_at=data.get("created_at", ""),
        questions=questions,
        alpha=data.get("alpha", 0.4),
    )


def list_exams() -> List[Dict]:
    """List all saved exam templates."""
    exams = []
    for path in sorted(EXAMS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            exams.append({
                "exam_id": data["exam_id"],
                "title": data["title"],
                "subject": data.get("subject", ""),
                "num_questions": len(data.get("questions", [])),
                "created_at": data.get("created_at", ""),
            })
        except Exception:
            continue
    return exams


def delete_exam(exam_id: str) -> bool:
    """Delete an exam template."""
    path = EXAMS_DIR / f"{exam_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def create_demo_exam() -> ExamTemplate:
    """Create a demo exam for testing."""
    exam = ExamTemplate(
        exam_id="demo_biology",
        title="Biology Midterm — Chapter 3",
        subject="Biology",
        alpha=0.4,
        questions=[
            Question(
                number=1,
                question_text="What is photosynthesis?",
                reference_answer=(
                    "Photosynthesis is the process by which green plants and some other "
                    "organisms use sunlight to synthesize nutrients from carbon dioxide "
                    "and water. It generally involves the green pigment chlorophyll and "
                    "generates oxygen as a by-product."
                ),
                max_score=5.0,
            ),
            Question(
                number=2,
                question_text="What is the role of mitochondria in a cell?",
                reference_answer=(
                    "The mitochondria is the organelle responsible for producing ATP "
                    "through the process of cellular respiration. It is often called "
                    "the powerhouse of the cell because it generates most of the cell's "
                    "supply of adenosine triphosphate (ATP), used as a source of chemical energy."
                ),
                max_score=5.0,
            ),
            Question(
                number=3,
                question_text="Explain the process of osmosis.",
                reference_answer=(
                    "Osmosis is the movement of water molecules through a semipermeable "
                    "membrane from a region of lower solute concentration to a region of "
                    "higher solute concentration, until equilibrium is reached."
                ),
                max_score=5.0,
            ),
            Question(
                number=4,
                question_text="What is DNA replication?",
                reference_answer=(
                    "DNA replication is a semiconservative process where the double helix "
                    "unwinds and each strand serves as a template for a new complementary "
                    "strand, resulting in two identical DNA molecules. Key enzymes include "
                    "helicase, primase, and DNA polymerase."
                ),
                max_score=5.0,
            ),
        ],
    )
    save_exam(exam)
    return exam
