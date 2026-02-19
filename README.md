# 🧠 AI Exam Corrector

An intelligent, end-to-end exam grading system that combines **OCR**, **BERT semantic similarity**, and **LLM reasoning** to automatically grade handwritten or typed student exam papers — with constructive feedback.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Datasets](#datasets)
- [How Grading Works](#how-grading-works)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Traditional exam grading is time-consuming and subjective. **AI Exam Corrector** automates this process using a hybrid approach:

1. **OCR** extracts student answers from scanned exam sheets (images or PDFs).
2. **BERT embeddings** measure semantic similarity between student and reference answers.
3. An **LLM** (Groq / Gemini / OpenRouter) applies expert-level reasoning to assign a score and generate constructive feedback.
4. A **hybrid formula** combines both signals for a balanced, fair final grade.

Teachers only need to configure reference answers once — after that, grading is fully automatic.

---

## Features

- **OCR Support** — EasyOCR (default) and Tesseract backends for extracting text from images and PDFs.
- **BERT Semantic Scoring** — Uses sentence-transformers (`all-MiniLM-L6-v2` by default) for cosine similarity matching.
- **LLM Grading & Feedback** — Leverages free LLM APIs (Groq, Google Gemini, OpenRouter) for nuanced scoring and written feedback.
- **Hybrid Scoring Formula** — `final_score = α × BERT_score + (1 − α) × LLM_score` with tunable α.
- **Streamlit Web UI** — Beautiful, dark-themed interface with two modes:
  - **Grade Mode** — Upload an exam image and get instant AI-powered scores.
  - **Setup Mode** — Configure exam templates with reference answers.
- **Exam Template Management** — Save/load exam configurations as JSON for reuse.
- **Dataset Evaluation** — Built-in loaders for standard ASAG benchmarks (Mohler, SemEval-2013, ASAP-SAS) with correlation metrics.
- **Detailed Analytics** — Per-question breakdowns, score distributions, grade letters, and exportable reports.

---

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────────┐
│  Exam Image/PDF  │────▶│  OCR Engine  │────▶│  Extracted Answers  │
└─────────────────┘     │ (EasyOCR /   │     └────────┬───────────┘
                        │  Tesseract)  │              │
                        └──────────────┘              ▼
                                            ┌─────────────────────┐
┌─────────────────┐                         │   Hybrid Grader     │
│ Exam Template   │────────────────────────▶│                     │
│ (ref answers)   │                         │  ┌───────────────┐  │
└─────────────────┘                         │  │ BERT Scorer   │  │
                                            │  │ (cosine sim)  │  │
                                            │  └───────┬───────┘  │
                                            │          │          │
                                            │  ┌───────▼───────┐  │
                                            │  │ LLM Feedback  │  │
                                            │  │ (score + text) │  │
                                            │  └───────┬───────┘  │
                                            │          │          │
                                            │  α·BERT + (1-α)·LLM │
                                            └──────────┬──────────┘
                                                       ▼
                                            ┌─────────────────────┐
                                            │   Final Grade +     │
                                            │   Feedback Report   │
                                            └─────────────────────┘
```

---

## Project Structure

```
AI_Correcting/
├── app.py               # Streamlit web UI (main entry point)
├── config.py            # Central configuration (API keys, model settings)
├── ocr_engine.py        # OCR backends (EasyOCR, Tesseract)
├── bert_scorer.py       # BERT semantic similarity scoring
├── llm_feedback.py      # LLM-based grading via Groq / Gemini / OpenRouter
├── hybrid_grader.py     # Hybrid grading pipeline (BERT + LLM)
├── exam_manager.py      # Exam template CRUD (JSON-based)
├── dataset_loader.py    # Standard ASAG dataset loaders & evaluation
├── test_pipeline.py     # End-to-end pipeline test script
├── report.tex           # LaTeX project report
├── requirements.txt     # Python dependencies
├── exams/               # Saved exam templates (JSON)
│   └── demo_biology.json
├── datasets/            # Downloaded evaluation datasets
│   └── README.md
└── samples/             # Sample exam images for testing
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- A free API key from at least one LLM provider:
  - [Groq](https://console.groq.com/) (recommended — fast & free)
  - [Google Gemini](https://aistudio.google.com/app/apikey)
  - [OpenRouter](https://openrouter.ai/)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI_Correcting.git
   cd AI_Correcting
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```env
   # LLM Provider: "groq", "gemini", or "openrouter"
   LLM_PROVIDER=groq

   # API Keys (add at least one)
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here

   # Optional tuning
   ALPHA=0.4
   BERT_MODEL_NAME=all-MiniLM-L6-v2
   OCR_ENGINE=easyocr
   OCR_LANGUAGES=en,fr
   ```

---

## Configuration

All settings are managed in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | LLM backend (`groq`, `gemini`, `openrouter`) |
| `GROQ_API_KEY` | — | Groq API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `ALPHA` | `0.4` | Hybrid weight: 0 = pure LLM, 1 = pure BERT |
| `BERT_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `OCR_ENGINE` | `easyocr` | OCR backend (`easyocr`, `tesseract`) |
| `OCR_LANGUAGES` | `en,fr` | OCR language codes |
| `MAX_SCORE` | `20` | Default maximum score |
| `LLM_TEMPERATURE` | `0.2` | LLM generation temperature |

---

## Usage

### Launch the Web UI

```bash
streamlit run app.py
```

This opens a Streamlit dashboard with two modes:

- **Grade Mode** — Upload a scanned exam image or PDF to get instant AI-generated grades and feedback.
- **Setup Mode** — Create and manage exam templates by defining questions and reference answers.

### Run the Test Pipeline

```bash
python test_pipeline.py
```

Runs built-in sample Q&A pairs through the full BERT + LLM hybrid grading pipeline and prints detailed results.

---

## Datasets

The system supports evaluation on standard Automated Short Answer Grading (ASAG) benchmarks:

| Dataset | Size | Score Range | Source |
|---|---|---|---|
| **Mohler et al. (2011)** | 2,273 answers | 0–5 | [GitHub](https://github.com/lm-pub-quiz/Mohler-dataset) |
| **SemEval-2013 Task 7** | Beetle + SciEntsBank | 5-way labels | [York CS](https://www.cs.york.ac.uk/semeval-2013/task7/) |
| **ASAP-SAS** | ~17k answers | 0–2 / 0–3 | [Kaggle](https://www.kaggle.com/c/asap-sas/data) |

See [datasets/README.md](datasets/README.md) for download and setup instructions.

---

## How Grading Works

1. **OCR Extraction** — The uploaded exam image/PDF is processed by EasyOCR (or Tesseract) to extract student answers as text.

2. **BERT Semantic Similarity** — Each student answer is encoded alongside the reference answer using a sentence-transformers model. Cosine similarity is computed and normalized to the score scale.

3. **LLM Expert Grading** — The reference answer, student answer, and BERT similarity are sent to an LLM which returns a numeric score and constructive feedback. The LLM considers:
   - Key concept coverage
   - Factual accuracy
   - Partial credit for incomplete answers

4. **Hybrid Score** — The final score combines both signals:
   ```
   final_score = α × BERT_normalized_score + (1 − α) × LLM_score
   ```
   Where α (default 0.4) balances semantic matching with expert reasoning.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
