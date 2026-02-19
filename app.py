"""
app.py — Streamlit UI for the AI Exam Correcting System.

Two modes:
  1. Grade  — Upload exam image → instant AI scores (no typing needed)
  2. Setup  — Configure exam templates (reference answers, one-time)

Launch:
    streamlit run app.py
"""

import io
import time
import json
import logging
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

import config
import exam_manager
from exam_manager import ExamTemplate, Question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Page Config & Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Exam Corrector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
    }

    /* ── Header ─────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #818cf8, #c084fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* ── Cards ──────────────────────────────────────── */
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: border-color 0.3s ease, transform 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99,102,241,0.3);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        font-weight: 600;
    }

    /* ── Grade Colors ───────────────────────────────── */
    .grade-excellent { color: #34d399; }
    .grade-good { color: #60a5fa; }
    .grade-average { color: #fbbf24; }
    .grade-poor { color: #f87171; }

    /* ── Score Badge ────────────────────────────────── */
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 700;
    }
    .badge-excellent { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
    .badge-good { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }
    .badge-average { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
    .badge-poor { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

    /* ── Feedback Card ──────────────────────────────── */
    .feedback-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-left: 4px solid #818cf8;
    }
    .feedback-card p {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.7;
        margin: 0;
    }

    /* ── Question Result Row ────────────────────────── */
    .q-result {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s;
    }
    .q-result:hover {
        border-color: rgba(99,102,241,0.25);
    }
    .q-number {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        text-align: center;
        border-radius: 8px;
        background: rgba(99,102,241,0.2);
        color: #818cf8;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 0.75rem;
    }

    /* ── Sidebar ────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Tweak Streamlit widgets ────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #6d28d9) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(79,70,229,0.3) !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(99,102,241,0.05);
        border: 2px dashed rgba(99,102,241,0.25);
        border-radius: 16px;
        padding: 1rem;
    }

    .stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: rgba(15,23,42,0.8) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Fix overlapping text ───────────────────────── */
    .stExpander summary span,
    .stExpander summary p,
    .stExpander [data-testid="stMarkdownContainer"] p {
        position: relative !important;
        z-index: 1 !important;
        letter-spacing: normal !important;
        word-spacing: normal !important;
    }
    .stExpander summary {
        overflow: visible !important;
        white-space: normal !important;
    }
    /* Ensure labels don't overlap */
    label[data-testid="stWidgetLabel"] {
        position: relative !important;
        display: block !important;
        overflow: visible !important;
    }
    .stSelectbox label, .stFileUploader label {
        letter-spacing: normal !important;
        font-kerning: normal !important;
    }
    /* Fix q-result card positioning */
    .q-result * {
        position: relative !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cached model loaders
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_bert():
    """Load BERT scorer (cached across reruns)."""
    import bert_scorer
    bert_scorer._get_model()  # Force model download
    return bert_scorer

@st.cache_resource(show_spinner=False)
def load_ocr():
    """Load OCR engine (cached)."""
    import ocr_engine
    return ocr_engine

@st.cache_resource(show_spinner=False)
def load_llm():
    import llm_feedback
    return llm_feedback

@st.cache_resource(show_spinner=False)
def load_grader():
    import hybrid_grader
    return hybrid_grader


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_grade_info(score, max_score):
    pct = (score / max_score) * 100 if max_score else 0
    if pct >= 85:
        return "Excellent", "excellent", "A+", "#34d399"
    if pct >= 70:
        return "Good", "good", "B+", "#60a5fa"
    if pct >= 50:
        return "Average", "average", "C", "#fbbf24"
    return "Needs Work", "poor", "D", "#f87171"


def make_gauge(score, max_score, title=""):
    pct = (score / max_score) * 100 if max_score else 0
    _, cls, _, color = get_grade_info(score, max_score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(15,23,42,0.5)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(248,113,113,0.08)"},
                {"range": [50, 70], "color": "rgba(251,191,36,0.08)"},
                {"range": [70, 85], "color": "rgba(96,165,250,0.08)"},
                {"range": [85, 100], "color": "rgba(52,211,153,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


def make_bar_chart(results):
    """Per-question score bar chart."""
    qs = [f"Q{r.question_id}" for r in results]
    bert_scores = [r.bert_result.normalised_score for r in results]
    llm_scores = [r.llm_result.score for r in results]
    final_scores = [r.final_score for r in results]
    max_scores = [r.max_score for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"BERT (a={results[0].alpha})",
        x=qs, y=bert_scores,
        marker_color="rgba(99,102,241,0.6)",
        marker_line=dict(color="#818cf8", width=1),
    ))
    fig.add_trace(go.Bar(
        name=f"LLM (1-a={1-results[0].alpha:.1f})",
        x=qs, y=llm_scores,
        marker_color="rgba(139,92,246,0.6)",
        marker_line=dict(color="#c084fc", width=1),
    ))
    fig.add_trace(go.Scatter(
        name="Final Score",
        x=qs, y=final_scores,
        mode="lines+markers",
        line=dict(color="#34d399", width=3),
        marker=dict(size=10, color="#34d399"),
    ))
    fig.add_trace(go.Scatter(
        name="Max Score",
        x=qs, y=max_scores,
        mode="lines",
        line=dict(color="#475569", width=1, dash="dash"),
    ))

    fig.update_layout(
        barmode="group",
        height=350,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0", "size": 12},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Score"),
    )
    return fig


def make_radar_chart(results):
    """Radar chart showing BERT vs LLM per question."""
    categories = [f"Q{r.question_id}" for r in results]
    bert_pcts = [(r.bert_result.normalised_score / r.max_score) * 100 for r in results]
    llm_pcts = [(r.llm_result.score / r.max_score) * 100 for r in results]
    # Close the polygon
    categories += [categories[0]]
    bert_pcts += [bert_pcts[0]]
    llm_pcts += [llm_pcts[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=bert_pcts, theta=categories, fill='toself',
        name='BERT', line_color='#818cf8',
        fillcolor='rgba(129,140,248,0.15)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=llm_pcts, theta=categories, fill='toself',
        name='LLM', line_color='#c084fc',
        fillcolor='rgba(192,132,252,0.15)',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        ),
        height=350,
        margin=dict(l=60, r=60, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0", "size": 12},
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        showlegend=True,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Grade an uploaded exam
# ═══════════════════════════════════════════════════════════════════════════════

def grade_uploaded_exam(uploaded_file, exam: ExamTemplate):
    """Full pipeline: Image -> OCR -> Segment -> BERT + LLM -> Results."""

    ocr = load_ocr()
    grader = load_grader()

    progress_bar = st.progress(0, text="Initializing OCR engine...")

    # ── Save uploaded file to temp ────────────────────────────────────────────
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # ── Step 1: OCR ──────────────────────────────────────────────────────────
    progress_bar.progress(10, text="Running OCR — extracting text from image...")
    raw_text = ocr.extract_text(tmp_path, preprocess=True)

    progress_bar.progress(30, text="Segmenting student answers...")
    student_answers_raw = ocr.segment_answers(raw_text)
    student_answers = {qid: ans for qid, ans in student_answers_raw}

    # ── Match student answers to exam questions ──────────────────────────────
    progress_bar.progress(40, text="Matching answers to questions...")

    pairs = []
    for q in exam.questions:
        qnum = str(q.number)
        # Try to find matching student answer by question number
        stu_ans = student_answers.get(qnum, "")
        if not stu_ans and len(student_answers_raw) >= q.number:
            # fallback: use positional match
            stu_ans = student_answers_raw[q.number - 1][1] if q.number - 1 < len(student_answers_raw) else ""
        pairs.append((qnum, q.reference_answer, stu_ans if stu_ans else "[No answer detected]"))

    # ── Step 2: Hybrid Grading ───────────────────────────────────────────────
    results = []
    total_questions = len(pairs)
    for i, (qid, ref, stu) in enumerate(pairs):
        pct = 40 + int((i / total_questions) * 55)
        q_obj = exam.questions[i]
        progress_bar.progress(pct, text=f"Grading Q{qid}: BERT + LLM analysis...")

        result = grader.grade_answer(
            reference=ref,
            student=stu,
            question_id=qid,
            max_score=q_obj.max_score,
            alpha=exam.alpha,
        )
        results.append(result)

    progress_bar.progress(100, text="Grading complete!")
    time.sleep(0.5)
    progress_bar.empty()

    return results, raw_text


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 2rem;">🧠</span>
        <h2 style="background: linear-gradient(135deg, #818cf8, #c084fc);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-weight: 800; margin: 0.25rem 0;">GradeAI</h2>
        <p style="color: #64748b; font-size: 0.75rem; letter-spacing: 0.1em;">
            HYBRID BERT + LLM ENGINE</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["Grade Exam", "Setup Exam", "Settings"],
        label_visibility="collapsed",
    )

    st.divider()

    # Quick info
    st.markdown(f"""
    <div style="padding: 0.75rem; background: rgba(15,23,42,0.5);
                border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;">
        <p style="color: #64748b; font-size: 0.7rem; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.1em;">Pipeline</p>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.25rem 0;">
            BERT: <code style="color: #818cf8;">{config.BERT_MODEL_NAME.split('/')[-1]}</code></p>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.25rem 0;">
            LLM: <code style="color: #c084fc;">{config.LLM_PROVIDER}</code></p>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.25rem 0;">
            Alpha: <code style="color: #fbbf24;">{config.ALPHA}</code></p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Grade Exam
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Grade Exam":
    st.markdown("""
    <div class="main-header">
        <h1>Upload & Grade</h1>
        <p>Upload a student's exam image — get instant AI-powered scores & feedback</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Select exam template ─────────────────────────────────────────────────
    exams_list = exam_manager.list_exams()

    if not exams_list:
        st.info("No exam templates found. Creating a demo exam for you...")
        exam_manager.create_demo_exam()
        exams_list = exam_manager.list_exams()

    exam_options = {f"{e['title']}  ({e['num_questions']} Qs)": e["exam_id"] for e in exams_list}

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_label = st.selectbox(
            "Select Exam Template",
            options=list(exam_options.keys()),
            help="Choose which exam's reference answers to grade against"
        )

    selected_exam_id = exam_options[selected_label]
    exam = exam_manager.load_exam(selected_exam_id)

    with col2:
        st.metric("Questions", len(exam.questions))

    # Show exam details in expander
    with st.expander(f"View reference answers for: {exam.title}", expanded=False):
        for q in exam.questions:
            st.markdown(f"""
            <div class="q-result">
                <span class="q-number">{q.number}</span>
                <strong style="color: #e2e8f0;">{q.question_text}</strong>
                <span style="float: right; color: #818cf8; font-weight: 600;">{q.max_score} pts</span>
                <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem; padding-left: 2.5rem;">
                    {q.reference_answer}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Upload area ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Student Exam",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
        help="Supported: images (PNG, JPG, TIFF, BMP) and PDF files",
    )

    if uploaded:
        # Show image preview
        try:
            if uploaded.name.lower().endswith(".pdf"):
                st.info(f"PDF uploaded: **{uploaded.name}** ({uploaded.size / 1024:.0f} KB)")
            else:
                img = Image.open(uploaded)
                st.image(img, caption=f"{uploaded.name}", use_container_width=True)
                uploaded.seek(0)  # Reset after preview
        except Exception:
            pass

        # ── Grade Button ─────────────────────────────────────────────────────
        if st.button("Grade This Exam", use_container_width=True, type="primary"):

            with st.spinner(""):
                t0 = time.time()
                results, ocr_text = grade_uploaded_exam(uploaded, exam)
                elapsed = time.time() - t0

            # ═══════════════════════════════════════════════════════════════════
            #  RESULTS DISPLAY
            # ═══════════════════════════════════════════════════════════════════

            st.markdown("---")

            # ── Summary Metrics ───────────────────────────────────────────────
            total_score = sum(r.final_score for r in results)
            total_max = sum(r.max_score for r in results)
            avg_cosine = sum(r.bert_result.cosine_similarity for r in results) / len(results)
            overall_pct = (total_score / total_max) * 100 if total_max else 0

            label, cls, letter, color = get_grade_info(total_score, total_max)

            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: rgba(15,23,42,0.6);
                        border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; margin-bottom: 2rem;">
                <p style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.5rem;">
                    OVERALL RESULT</p>
                <div style="font-size: 3.5rem; font-weight: 900; color: {color}; line-height: 1;">
                    {total_score:.1f}<span style="font-size: 1.5rem; opacity: 0.5;">/{total_max:.0f}</span>
                </div>
                <div class="score-badge badge-{cls}" style="margin-top: 0.75rem;">
                    {letter} — {label} — {overall_pct:.0f}%
                </div>
                <p style="color: #475569; font-size: 0.8rem; margin-top: 1rem;">
                    Graded in {elapsed:.1f}s  |  alpha={exam.alpha}  |  {len(results)} questions
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge + Radar Chart ───────────────────────────────────────────
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_gauge(total_score, total_max, "Overall Score"),
                               use_container_width=True)
            with c2:
                if len(results) >= 3:
                    st.plotly_chart(make_radar_chart(results), use_container_width=True)
                else:
                    st.plotly_chart(make_gauge(avg_cosine * total_max, total_max, "Avg Similarity"),
                                   use_container_width=True)

            # ── Bar Chart ─────────────────────────────────────────────────────
            st.plotly_chart(make_bar_chart(results), use_container_width=True)

            # ── Per-Question Details ──────────────────────────────────────────
            st.markdown("### Per-Question Breakdown")

            for r in results:
                q_idx = int(r.question_id) - 1
                q_obj = exam.questions[q_idx] if q_idx < len(exam.questions) else None
                q_label, q_cls, q_letter, q_color = get_grade_info(r.final_score, r.max_score)

                with st.container():
                    st.markdown(f"""
                    <div class="q-result">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <div>
                                <span class="q-number">{r.question_id}</span>
                                <strong style="color: #e2e8f0;">{q_obj.question_text if q_obj else f'Question {r.question_id}'}</strong>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 1.5rem; font-weight: 800; color: {q_color};">
                                    {r.final_score:.1f}</span>
                                <span style="color: #475569; font-size: 0.9rem;">/{r.max_score:.0f}</span>
                                <span class="score-badge badge-{q_cls}" style="margin-left: 0.5rem; font-size: 0.75rem;">
                                    {q_letter} {q_label}
                                </span>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; margin: 0.75rem 0;">
                            <div style="text-align: center; padding: 0.5rem; background: rgba(99,102,241,0.08); border-radius: 8px;">
                                <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">BERT Score</div>
                                <div style="color: #818cf8; font-weight: 700; font-size: 1.1rem;">{r.bert_result.normalised_score:.1f}</div>
                            </div>
                            <div style="text-align: center; padding: 0.5rem; background: rgba(139,92,246,0.08); border-radius: 8px;">
                                <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">LLM Score</div>
                                <div style="color: #c084fc; font-weight: 700; font-size: 1.1rem;">{r.llm_result.score:.1f}</div>
                            </div>
                            <div style="text-align: center; padding: 0.5rem; background: rgba(52,211,153,0.08); border-radius: 8px;">
                                <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">Cosine Sim</div>
                                <div style="color: #34d399; font-weight: 700; font-size: 1.1rem;">{r.bert_result.cosine_similarity:.3f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Feedback
                    st.markdown(f"""
                    <div class="feedback-card">
                        <p>{r.feedback}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show detected student answer
                    with st.expander(f"Detected student answer for Q{r.question_id}"):
                        st.text(r.student_answer)

            # ── OCR Text ──────────────────────────────────────────────────────
            with st.expander("View raw OCR output"):
                st.code(ocr_text, language=None)

            # ── Download results ──────────────────────────────────────────────
            results_data = {
                "exam": exam.title,
                "total_score": round(total_score, 1),
                "total_max": total_max,
                "percentage": round(overall_pct, 1),
                "alpha": exam.alpha,
                "questions": [
                    {
                        "number": r.question_id,
                        "final_score": r.final_score,
                        "bert_score": r.bert_result.normalised_score,
                        "llm_score": r.llm_result.score,
                        "cosine_similarity": r.bert_result.cosine_similarity,
                        "feedback": r.feedback,
                        "student_answer": r.student_answer,
                    }
                    for r in results
                ],
            }
            st.download_button(
                "Download Results (JSON)",
                data=json.dumps(results_data, indent=2, ensure_ascii=False),
                file_name=f"results_{exam.exam_id}.json",
                mime="application/json",
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Setup Exam
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Setup Exam":
    st.markdown("""
    <div class="main-header">
        <h1>Setup Exam Template</h1>
        <p>Configure reference answers once — then just upload student papers to grade</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Create New Exam", "Manage Exams"])

    with tab1:
        with st.form("new_exam_form"):
            c1, c2 = st.columns(2)
            with c1:
                exam_title = st.text_input("Exam Title", placeholder="e.g. Biology Midterm Ch.3")
            with c2:
                exam_subject = st.text_input("Subject", placeholder="e.g. Biology")

            c3, c4 = st.columns(2)
            with c3:
                exam_id = st.text_input("Exam ID (unique)", placeholder="e.g. bio_midterm_ch3")
            with c4:
                alpha_val = st.slider("Alpha (BERT weight)", 0.0, 1.0, 0.4, 0.05)

            num_questions = st.number_input("Number of questions", min_value=1, max_value=50, value=4)

            st.markdown("### Questions & Reference Answers")

            questions_data = []
            for i in range(int(num_questions)):
                st.markdown(f"**Question {i+1}**")
                c_q, c_s = st.columns([4, 1])
                with c_q:
                    q_text = st.text_input(f"Question text #{i+1}", key=f"qt_{i}",
                                           placeholder="What is photosynthesis?")
                with c_s:
                    q_score = st.number_input(f"Max pts #{i+1}", min_value=1.0, max_value=100.0,
                                              value=5.0, step=0.5, key=f"qs_{i}")
                ref_ans = st.text_area(f"Reference answer #{i+1}", key=f"ra_{i}", height=80,
                                        placeholder="The complete ideal answer...")
                questions_data.append((q_text, ref_ans, q_score))

            submitted = st.form_submit_button("Save Exam Template", use_container_width=True)

            if submitted:
                if not exam_id.strip() or not exam_title.strip():
                    st.error("Please fill in exam title and ID.")
                else:
                    questions = [
                        Question(number=i+1, question_text=qt, reference_answer=ra, max_score=qs)
                        for i, (qt, ra, qs) in enumerate(questions_data)
                        if ra.strip()
                    ]
                    if not questions:
                        st.error("Please add at least one reference answer.")
                    else:
                        exam_obj = ExamTemplate(
                            exam_id=exam_id.strip(),
                            title=exam_title.strip(),
                            subject=exam_subject.strip(),
                            questions=questions,
                            alpha=alpha_val,
                        )
                        exam_manager.save_exam(exam_obj)
                        st.success(f"Exam **{exam_title}** saved with {len(questions)} questions!")
                        st.balloons()

    with tab2:
        exams = exam_manager.list_exams()
        if not exams:
            st.info("No exams configured yet. Create one above!")
        else:
            for e in exams:
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(f"**{e['title']}** — {e['subject']} ({e['num_questions']} Qs)")
                with c2:
                    st.caption(e.get("created_at", "")[:10])
                with c3:
                    if st.button("Delete", key=f"del_{e['exam_id']}"):
                        exam_manager.delete_exam(e["exam_id"])
                        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Settings
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Settings":
    st.markdown("""
    <div class="main-header">
        <h1>System Settings</h1>
        <p>Configuration & pipeline information</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Current Configuration")
        st.markdown(f"""
        | Setting | Value |
        |---------|-------|
        | **LLM Provider** | `{config.LLM_PROVIDER}` |
        | **LLM Model** | `{config.LLM_MODELS.get(config.LLM_PROVIDER, 'N/A')}` |
        | **BERT Model** | `{config.BERT_MODEL_NAME}` |
        | **Alpha (BERT weight)** | `{config.ALPHA}` |
        | **OCR Engine** | `{config.OCR_ENGINE}` |
        | **OCR Languages** | `{config.OCR_LANGUAGES}` |
        | **Max Score** | `{config.MAX_SCORE}` |
        """)

    with c2:
        st.markdown("### Architecture")
        st.code("""
    Exam Image/PDF
         |
         v
    +-----------+
    |    OCR    |   (EasyOCR)
    +-----+-----+
          |
          v
    +---------------+
    | Answer Segm.  |
    +-----+---------+
          |
    +-----+-----------+
    |                  |
    v                  v
  BERT Scorer     LLM Grader
  (cosine sim)    (score+feedback)
    |                  |
    +------+----------+
           v
     final = a*BERT + (1-a)*LLM
        """, language=None)

    st.markdown("### Supported Datasets for Evaluation")
    st.markdown("""
    | # | Dataset | Samples | Link |
    |---|---------|---------|------|
    | 1 | **Mohler et al. (2011)** | 2,273 | [GitHub](https://github.com/lm-pub-quiz/Mohler-dataset) |
    | 2 | **SemEval-2013 Task 7** | Beetle + SciEntsBank | [HuggingFace](https://huggingface.co/datasets/semeval2013_task7) |
    | 3 | **ASAP-SAS** | ~17k | [Kaggle](https://www.kaggle.com/c/asap-sas/data) |
    """)

    # Test API connection
    st.markdown("### API Connection Test")
    if st.button("Test Groq API"):
        try:
            key = config.get_api_key()
            import requests
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if resp.ok:
                models = [m["id"] for m in resp.json().get("data", [])][:5]
                st.success(f"Connected! Available models: {', '.join(models)}")
            else:
                st.error(f"API returned {resp.status_code}")
        except Exception as e:
            st.error(f"{e}")
