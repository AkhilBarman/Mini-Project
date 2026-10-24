"""
Legal Hallucination Detector - Modern Interactive Web Platform
Phase 5: Benchmark Evaluation & Real-Time Inspection

Benchmarking Cross-Encoder NLI (SummaC) vs. Black-Box Consistency (SelfCheckGPT)
on Indian Supreme Court Judgments (MILDSum Benchmark).
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# Ensure src is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hd.text_utils import clean_text, split_sentences
from hd.comparison import calibrate_score, compute_hybrid_scores


# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Hallucination Detector | MILDSum Benchmark",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# Version-Agnostic Safe Display Helpers
# -----------------------------------------------------------------------------
def safe_render_image(image_path: str, caption: str = ""):
    """Safely renders an image regardless of Streamlit version argument support."""
    if not os.path.exists(image_path):
        st.info(f"Image not found at: {image_path}")
        return
    try:
        st.image(image_path, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(image_path, caption=caption, use_column_width=True)
        except TypeError:
            st.image(image_path, caption=caption)


def safe_render_dataframe(df: pd.DataFrame):
    """Safely renders a dataframe across all Streamlit versions."""
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        try:
            st.dataframe(df, use_container_width=True)
        except TypeError:
            st.dataframe(df)


# -----------------------------------------------------------------------------
# Premium Clean CSS (Linear / Vercel Modern Dark Theme)
# -----------------------------------------------------------------------------
MODERN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Font Reset */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1360px !important;
    }

    /* Top Executive Header */
    .top-header {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    .top-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-bottom: 10px;
    }
    .header-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0 0 6px 0;
        letter-spacing: -0.025em;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* Modern KPI Metric Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 18px 20px;
        backdrop-filter: blur(8px);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #64748b;
    }
    .kpi-card.blue::before { background: #38bdf8; }
    .kpi-card.green::before { background: #10b981; }
    .kpi-card.rose::before { background: #f43f5e; }
    .kpi-card.purple::before { background: #a855f7; }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #64748b;
    }

    /* Sentence Cards - Clean, Spacious, Highly Readable */
    .sentence-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: all 0.15s ease-in-out;
    }
    .sentence-box:hover {
        background: rgba(30, 41, 59, 0.65);
        border-color: rgba(148, 163, 184, 0.25);
    }
    .sentence-box.supported {
        border-left: 4px solid #10b981;
    }
    .sentence-box.hallucinated {
        border-left: 4px solid #f43f5e;
        background: rgba(244, 63, 94, 0.04);
    }

    .sentence-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .sentence-num {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .tag-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .tag-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .tag-red {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
    }
    .tag-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        margin-left: 6px;
    }
    .score-txt {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #cbd5e1;
    }

    .sentence-text {
        font-size: 0.96rem;
        color: #f1f5f9;
        line-height: 1.6;
        margin-bottom: 6px;
    }

    /* Verification Diff Box */
    .diff-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed rgba(244, 63, 94, 0.35);
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 10px;
        font-size: 0.86rem;
    }
    .diff-row {
        margin-bottom: 6px;
        line-height: 1.45;
    }
    .diff-row:last-child {
        margin-bottom: 0;
    }
    .diff-label {
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .diff-error { color: #fb7185; }
    .diff-truth { color: #34d399; }
    .diff-info { color: #38bdf8; }

    /* Comparison Head-to-Head Banner */
    .consensus-banner {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-around;
        text-align: center;
    }
    .consensus-item-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
    }
    .consensus-item-val {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 2px;
    }

    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 8px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.1) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
    }
</style>
"""
st.markdown(MODERN_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
@st.cache_data
def load_app_data():
    """Load testset records, predictions, sample judgments, and comparison summary."""
    testset_path = os.path.join(BASE_DIR, "data", "testset", "testset.csv")
    pairs_path = os.path.join(BASE_DIR, "data", "processed", "sample_pairs.jsonl")
    pred_a_path = os.path.join(BASE_DIR, "results", "approachA_mnli_base_predictions.csv")
    pred_b_path = os.path.join(BASE_DIR, "results", "approachB_selfcheck_predictions.csv")
    pred_h_path = os.path.join(BASE_DIR, "results", "approach_hybrid_predictions.csv")
    comp_path = os.path.join(BASE_DIR, "results", "comparison_summary.csv")

    testset_df = pd.read_csv(testset_path) if os.path.exists(testset_path) else pd.DataFrame()
    pred_a = pd.read_csv(pred_a_path) if os.path.exists(pred_a_path) else pd.DataFrame()
    pred_b = pd.read_csv(pred_b_path) if os.path.exists(pred_b_path) else pd.DataFrame()
    pred_h = pd.read_csv(pred_h_path) if os.path.exists(pred_h_path) else pd.DataFrame()
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()

    # Load judgment texts
    judgments_map = {}
    if os.path.exists(pairs_path):
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pid = rec.get("id")
                if pid:
                    judgments_map[pid] = rec

    merged = testset_df.copy()
    if not pred_a.empty:
        merged = merged.merge(pred_a[["row_id", "hallucination_score"]].rename(columns={"hallucination_score": "score_a"}), on="row_id", how="left")
    if not pred_b.empty:
        merged = merged.merge(pred_b[["row_id", "hallucination_score"]].rename(columns={"hallucination_score": "score_b"}), on="row_id", how="left")
    if not pred_h.empty:
        merged = merged.merge(pred_h[["row_id", "hallucination_score"]].rename(columns={"hallucination_score": "score_hybrid"}), on="row_id", how="left")

    return {
        "merged_df": merged,
        "judgments_map": judgments_map,
        "comp_df": comp_df
    }


DATA = load_app_data()
merged_df = DATA["merged_df"]
judgments_map = DATA["judgments_map"]
comp_df = DATA["comp_df"]


# -----------------------------------------------------------------------------
# Top Executive Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="top-header">
        <div class="top-badge">⚖️ LegalNLP • B.Tech Mini-Project</div>
        <h1 class="header-title">Legal Hallucination Detection System</h1>
        <p class="header-subtitle">
            Sentence-level verification of AI-generated summaries for Indian Court Judgments (MILDSum Benchmark).
            Comparing <strong>Cross-Encoder NLI (SummaC)</strong> against <strong>Black-Box Consistency (SelfCheckGPT)</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Global Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Inspection Settings")

    model_option = st.selectbox(
        "Detection Model:",
        options=[
            "Approach A: SummaC NLI (Calibrated θ=0.85)",
            "Approach B: SelfCheckGPT (Consistency θ=0.50)",
            "Hybrid Ensemble (NLI + SelfCheck θ=0.50)"
        ],
        index=0,
        help="Choose the underlying hallucination detection backbone."
    )

    if "Approach A" in model_option:
        rec_theta = 0.85
        score_col = "score_a"
        model_name = "SummaC Cross-Encoder NLI"
    elif "Approach B" in model_option:
        rec_theta = 0.50
        score_col = "score_b"
        model_name = "SelfCheckGPT Consistency"
    else:
        rec_theta = 0.50
        score_col = "score_hybrid"
        model_name = "Calibrated Hybrid Ensemble"

    st.markdown("---")
    st.markdown("#### 🎯 Decision Threshold ($\theta$)")
    threshold = st.slider(
        "Flag Threshold:",
        min_value=0.10,
        max_value=0.95,
        value=rec_theta,
        step=0.01,
        help="Sentences scoring above this threshold are classified as hallucinations."
    )

    st.caption(f"Recommended for {model_name}: **θ = {rec_theta:.2f}**")

    st.markdown("---")
    st.markdown("#### 🔍 Filter Sentences")
    filter_mode = st.radio(
        "Display Filter:",
        options=["All Sentences", "⚠️ Flagged Hallucinations Only", "🎯 Injected Errors Only", "✅ Supported Only"],
        index=0
    )

    st.markdown("---")
    st.caption("Benchmark: 462 Sentences • 10 Indian Court Cases • 16 Human-Reviewed Legal Mutations")


# -----------------------------------------------------------------------------
# Primary Navigation Tabs (Concise, Uncluttered, Perfectly Aligned)
# -----------------------------------------------------------------------------
tab_review, tab_compare, tab_benchmark, tab_playground, tab_viva = st.tabs([
    "📋 Review Case",
    "⚖️ Compare Models",
    "📊 Benchmark Analytics",
    "🧪 Live Playground",
    "📖 Viva & Methodology"
])


# =============================================================================
# TAB 1: REVIEW CASE (CLEAN, FOCUSED SENTENCE REVIEW)
# =============================================================================
with tab_review:
    # 1. Top Case Control Bar
    bar_c1, bar_c2 = st.columns([2, 1])

    case_options = [f"Sample_{i}" for i in range(1, 11)]
    with bar_c1:
        selected_case = st.selectbox(
            "Select Legal Case (MILDSum Judgment):",
            options=case_options,
            index=0,
            help="Select one of 10 Indian Supreme Court judgment cases from the testset."
        )

    case_sentences = merged_df[merged_df["pair_id"] == selected_case].copy()
    case_judgment = judgments_map.get(selected_case, {})
    judgment_text = case_judgment.get("judgment_text", "Judgment text not loaded.")

    with bar_c2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        show_source = st.checkbox("📜 View Source Judgment Text", value=False)

    # Collapsible Source Judgment Viewer
    if show_source:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <strong style="color: #38bdf8;">Court Judgment: {selected_case}</strong>
                    <span style="font-size: 0.8rem; color: #94a3b8;">{len(judgment_text.split()):,} words</span>
                </div>
                <div style="max-height: 260px; overflow-y: auto; font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; white-space: pre-wrap; padding-right: 10px;">
                    {judgment_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. KPI Metrics Ribbon
    scores = case_sentences[score_col].values
    true_labels = case_sentences["label"].values
    pred_labels = (scores >= threshold).astype(int)

    n_total = len(case_sentences)
    n_flagged = int(np.sum(pred_labels))
    n_supported = n_total - n_flagged
    n_ground_truth_errors = int(np.sum(true_labels == 1))
    n_tp = int(np.sum((pred_labels == 1) & (true_labels == 1)))

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card blue">
                <div class="kpi-label">Summary Sentences</div>
                <div class="kpi-val">{n_total}</div>
                <div class="kpi-sub">Total claims in summary</div>
            </div>
            <div class="kpi-card green">
                <div class="kpi-label">Verified Supported</div>
                <div class="kpi-val" style="color: #34d399;">{n_supported}</div>
                <div class="kpi-sub">Score &lt; {threshold:.2f}</div>
            </div>
            <div class="kpi-card rose">
                <div class="kpi-label">Flagged Hallucinations</div>
                <div class="kpi-val" style="color: {'#fb7185' if n_flagged > 0 else '#94a3b8'};">{n_flagged}</div>
                <div class="kpi-sub">Requires verification</div>
            </div>
            <div class="kpi-card purple">
                <div class="kpi-label">Ground Truth Errors</div>
                <div class="kpi-val">{n_ground_truth_errors}</div>
                <div class="kpi-sub">{n_tp} of {n_ground_truth_errors} caught by model</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Filter Sentences
    filtered_indices = []
    for idx, (_, row) in enumerate(case_sentences.iterrows()):
        is_f = scores[idx] >= threshold
        is_err = int(row["label"]) == 1

        if filter_mode == "All Sentences":
            filtered_indices.append(idx)
        elif filter_mode == "⚠️ Flagged Hallucinations Only" and is_f:
            filtered_indices.append(idx)
        elif filter_mode == "🎯 Injected Errors Only" and is_err:
            filtered_indices.append(idx)
        elif filter_mode == "✅ Supported Only" and not is_f:
            filtered_indices.append(idx)

    # 4. Render Sentence Review Cards
    if not filtered_indices:
        st.info(f"No sentences match the filter '{filter_mode}' at threshold θ = {threshold:.2f}.")
    else:
        st.markdown(f"**Showing {len(filtered_indices)} of {n_total} summary sentences:**")

        for idx in filtered_indices:
            row = case_sentences.iloc[idx]
            score = float(scores[idx]) if not np.isnan(scores[idx]) else 0.0
            is_flagged = score >= threshold
            is_actual_hallucinated = int(row["label"]) == 1
            error_type = str(row.get("error_type", "")) if pd.notna(row.get("error_type")) else ""
            orig_sentence = str(row.get("original_sentence", "")) if pd.notna(row.get("original_sentence")) else ""

            card_class = "hallucinated" if is_flagged else "supported"
            status_tag = '<span class="tag-pill tag-red">⚠️ Hallucination</span>' if is_flagged else '<span class="tag-pill tag-green">✓ Supported</span>'
            error_tag = f'<span class="tag-pill tag-amber">{error_type}</span>' if error_type else ""

            # Diff block if this was a known injected error
            diff_html = ""
            if is_actual_hallucinated:
                diff_html = f"""
                <div class="diff-box">
                    <div class="diff-row"><span class="diff-label diff-error">❌ Generated Claim:</span> {row['sentence_text']}</div>
                    <div class="diff-row"><span class="diff-label diff-truth">✅ Factual Judgment Record:</span> {orig_sentence}</div>
                    <div class="diff-row"><span class="diff-label diff-info">💡 Error Category:</span> {error_type}</div>
                </div>
                """

            st.markdown(
                f"""
                <div class="sentence-box {card_class}">
                    <div class="sentence-header">
                        <div class="sentence-num">Sentence #{idx + 1} &nbsp; {status_tag} {error_tag}</div>
                        <div class="score-txt">Score: <strong>{score:.4f}</strong> (θ={threshold:.2f})</div>
                    </div>
                    <div class="sentence-text">{row['sentence_text']}</div>
                    {diff_html}
                </div>
                """,
                unsafe_allow_html=True
            )


# =============================================================================
# TAB 2: COMPARE MODELS (HEAD-TO-HEAD ANALYSIS)
# =============================================================================
with tab_compare:
    st.markdown("### ⚖️ Head-to-Head Model Comparison")
    st.caption(f"Compare **Approach A (SummaC NLI)** vs. **Approach B (SelfCheckGPT)** on **{selected_case}**.")

    case_sentences = merged_df[merged_df["pair_id"] == selected_case].copy()
    scores_a = case_sentences["score_a"].values
    scores_b = case_sentences["score_b"].values
    flags_a = (scores_a >= 0.85).astype(int)
    flags_b = (scores_b >= 0.50).astype(int)

    agreed = np.sum(flags_a == flags_b)
    total_sents = len(case_sentences)
    consensus_pct = (agreed / total_sents) * 100
    both_flagged = np.sum((flags_a == 1) & (flags_b == 1))
    a_only = np.sum((flags_a == 1) & (flags_b == 0))
    b_only = np.sum((flags_a == 0) & (flags_b == 1))

    # Consensus Ribbon
    st.markdown(
        f"""
        <div class="consensus-banner">
            <div>
                <div class="consensus-item-title">Agreement Rate</div>
                <div class="consensus-item-val" style="color: #38bdf8;">{consensus_pct:.1f}%</div>
                <div style="font-size: 0.75rem; color: #64748b;">{agreed} of {total_sents} match</div>
            </div>
            <div>
                <div class="consensus-item-title">Both Flagged</div>
                <div class="consensus-item-val" style="color: #fb7185;">{both_flagged}</div>
                <div style="font-size: 0.75rem; color: #64748b;">High confidence</div>
            </div>
            <div>
                <div class="consensus-item-title">Approach A Only</div>
                <div class="consensus-item-val" style="color: #fbbf24;">{a_only}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Logical contradictions</div>
            </div>
            <div>
                <div class="consensus-item-title">Approach B Only</div>
                <div class="consensus-item-val" style="color: #a855f7;">{b_only}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Low consistency</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    diff_only = st.checkbox("Show Only Sentences Where Models Disagree", value=False)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Approach A: SummaC NLI (θ=0.85)")
        st.caption("Cross-Encoder premise-hypothesis checking against judgment text.")

    with col_b:
        st.markdown("#### Approach B: SelfCheckGPT (θ=0.50)")
        st.caption("Stochastic self-consistency checking across 3 sampled summaries.")

    for idx, (_, row) in enumerate(case_sentences.iterrows()):
        f_a = flags_a[idx] == 1
        f_b = flags_b[idx] == 1

        if diff_only and (f_a == f_b):
            continue

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="sentence-box {'hallucinated' if f_a else 'supported'}" style="padding: 12px 16px;">
                    <div class="sentence-header">
                        <span class="sentence-num">#{idx+1}</span>
                        <span class="tag-pill {'tag-red' if f_a else 'tag-green'}">{'⚠️ Flagged' if f_a else '✓ Supported'}</span>
                    </div>
                    <div style="font-size: 0.88rem; color: #e2e8f0;">{row['sentence_text']}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 6px;">Score: {scores_a[idx]:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="sentence-box {'hallucinated' if f_b else 'supported'}" style="padding: 12px 16px;">
                    <div class="sentence-header">
                        <span class="sentence-num">#{idx+1}</span>
                        <span class="tag-pill {'tag-red' if f_b else 'tag-green'}">{'⚠️ Flagged' if f_b else '✓ Supported'}</span>
                    </div>
                    <div style="font-size: 0.88rem; color: #e2e8f0;">{row['sentence_text']}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 6px;">Score: {scores_b[idx]:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =============================================================================
# TAB 3: BENCHMARK ANALYTICS (METRICS, CHARTS, ROC CURVES)
# =============================================================================
with tab_benchmark:
    st.markdown("### 📊 Benchmark Performance & Research Analytics")
    st.caption("Evaluated on 462 sentences from 10 Indian court judgments (`data/testset/testset.csv`).")

    if not comp_df.empty:
        # Cleanly format metrics table
        clean_comp = comp_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg Precision"]:
            if col in clean_comp.columns:
                clean_comp[col] = clean_comp[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) and col in ["Accuracy", "Precision", "Recall"] else (f"{x:.4f}" if pd.notna(x) else "-"))
        
        safe_render_dataframe(clean_comp)
    else:
        st.info("Comparison table not found. Run scripts/compare_approaches.py")

    st.markdown("<br>", unsafe_allow_html=True)

    # Safe visual chart rendering
    chart_c1, chart_c2 = st.columns(2)

    chart_file = os.path.join(BASE_DIR, "results", "comparison_breakdown.png")
    roc_file = os.path.join(BASE_DIR, "results", "roc_curve_approach_a.png")

    with chart_c1:
        st.markdown("#### 📈 Multi-Model Comparative Evaluation")
        safe_render_image(chart_file, caption="Figure 1: Core Metrics, Latency Profile & Error Breakdown")

    with chart_c2:
        st.markdown("#### 🎯 Approach A Discriminative ROC Curve")
        safe_render_image(roc_file, caption="Figure 2: Receiver Operating Characteristic (ROC-AUC = 0.8585)")


# =============================================================================
# TAB 4: LIVE CUSTOM PLAYGROUND
# =============================================================================
with tab_playground:
    st.markdown("### 🧪 Live Hallucination Playground")
    st.caption("Paste any custom Indian court judgment and summary to detect hallucinations on the fly.")

    default_j = (
        "The appellant, Smt. Rekha Sharma, filed the present criminal appeal against the judgment of the High Court of Delhi "
        "dated 14th August 2018. The High Court had upheld the conviction of the appellant under Section 302 read with Section 34 "
        "of the Indian Penal Code, sentencing her to life imprisonment and a fine of Rs. 25,000. After examining the testimony of PW-1 "
        "and ballistic forensic reports, this Court finds significant material contradictions in the prosecution story. Consequently, "
        "the conviction cannot be sustained. The appeal is allowed and the appellant is acquitted of all charges. The appellant shall "
        "be released forthwith unless required in any other case."
    )

    default_s = (
        "The Supreme Court heard an appeal by Smt. Rekha Sharma challenging her conviction under Section 302 IPC.\n"
        "The High Court of Delhi had previously sentenced her to life imprisonment with a fine of Rs. 25,000.\n"
        "The Supreme Court dismissed the appeal and confirmed the death sentence of the appellant.\n"
        "The Court also directed the state to pay compensation of Rs. 50,00,000 to the victim's family."
    )

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        user_j = st.text_area("Judgment Source Text:", value=default_j, height=220)
    with col_in2:
        user_s = st.text_area("Candidate Summary Text:", value=default_s, height=220)

    if st.button("🚀 Analyze Summary for Hallucinations", type="primary"):
        with st.spinner("Analyzing claims against judgment..."):
            cleaned = clean_text(user_s)
            sents = split_sentences(cleaned)

            st.markdown(f"**Found {len(sents)} claims in summary:**")

            from hd.approach_b import compute_token_containment

            for i, sent in enumerate(sents):
                containment = compute_token_containment(sent, user_j)
                h_score = round(1.0 - containment, 4)
                is_h = h_score >= 0.50

                card_class = "hallucinated" if is_h else "supported"
                badge = '<span class="tag-pill tag-red">⚠️ Hallucination</span>' if is_h else '<span class="tag-pill tag-green">✓ Supported</span>'

                st.markdown(
                    f"""
                    <div class="sentence-box {card_class}">
                        <div class="sentence-header">
                            <span class="sentence-num">Sentence #{i+1} &nbsp; {badge}</span>
                            <span class="score-txt">Consistency: <strong>{containment*100:.1f}%</strong> | Risk: <strong>{h_score:.4f}</strong></span>
                        </div>
                        <div class="sentence-text">{sent}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =============================================================================
# TAB 5: VIVA & METHODOLOGY (CLEAR VIVA DEFENSE GUIDE)
# =============================================================================
with tab_viva:
    st.markdown("### 📖 Scientific Methodology & Viva Defense Guide")
    st.caption("Concise theoretical foundations and model answers for student viva presentation.")

    with st.expander("Q1: Why compare NLI with Self-Consistency? What makes them fundamentally different?", expanded=True):
        st.markdown(
            """
            - **Approach A (Cross-Encoder NLI / SummaC):** **Extrinsic reference-based** method. Directly compares the summary sentence against premise chunks of the judgment using a deep neural network (`microsoft/deberta-base-mnli`).
            - **Approach B (Self-Consistency / SelfCheckGPT):** **Intrinsic reference-free (black-box)** method. Checks whether facts generated by the LLM survive across multiple stochastic summary passes ($N=3$).
            - **Key Scientific Finding:** They are **complementary**: Approach A achieves **100% recall** on subtle legal flips (dates, numbers, outcomes), while Approach B runs **in 0.50s on CPU** with **92.4% accuracy** and cleanly eliminates pure fabrications.
            """
        )

    with st.expander("Q2: Why did Approach A require domain calibration (θ = 0.85 instead of 0.50)?", expanded=True):
        st.markdown(
            """
            General NLI models are trained on short, conversational sentences (MNLI). Indian court judgments feature dense, multi-clause syntax with archaic legal jargon. 
            Because NLI cross-encoders classify stylistic mismatches as *neutral* rather than *entailment*, uncalibrated support scores cluster around 0.65–0.75 even for true facts. 
            Adjusting $\theta$ from 0.50 to **0.85** eliminates 62% of false alarms, raising accuracy from 27.3% to **72.3%** while maintaining **100% recall**.
            """
        )

    with st.expander("Q3: Why can Approach B catch 100% of fabrications, but misses single-word mutations?", expanded=True):
        st.markdown(
            """
            - **Fabrications:** Entirely invented procedural orders or directives do not exist in the judgment and never reappear across sampled summaries, resulting in 0% containment ($H = 1.0$).
            - **Single-Word Mutations:** When an outcome or date is changed (e.g. *granted* to *dismissed*), 90%+ of sentence tokens remain identical. Lexical token containment retains high overlap, meaning black-box consensus cannot catch the contradiction without an NLI classifier.
            """
        )

    with st.expander("Q4: What is the recommended production architecture for legal AI applications?", expanded=True):
        st.markdown(
            """
            A **two-tiered hybrid architecture**:
            1. **Tier 1 (Fast Filter):** Run **Approach B (SelfCheckGPT)** first on CPU in <1s. Flag and reject any summary with low consensus.
            2. **Tier 2 (Deep Auditor):** Run **Approach A (SummaC Cross-Encoder)** on the critical operative clauses (dates, parties, outcomes, compensation figures) to guarantee zero factual flips.
            """
        )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.82rem;">
        Legal Hallucination Detection Platform • B.Tech Sem 5 Mini-Project (Deep Learning + NLP) • 
        Built with Streamlit, PyTorch, Transformers, SummaC & SelfCheckGPT
    </div>
    """,
    unsafe_allow_html=True
)
