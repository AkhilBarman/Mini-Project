"""
Legal Hallucination Detector - Interactive Web Demo
Phase 5: Final Demonstration & Comparative Evaluation Platform

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
# Streamlit App Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Hallucination Detector | MILDSum Benchmark",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Dark Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        color: #0f172a;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* Metrics Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f1f5f9;
        margin-bottom: 2px;
    }
    .metric-caption {
        font-size: 0.75rem;
        color: #64748b;
    }

    /* Sentence Cards */
    .sentence-card {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-left: 5px solid #64748b;
        transition: all 0.2s ease;
    }
    .sentence-card:hover {
        background: rgba(30, 41, 59, 0.8);
    }
    .sentence-card-supported {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.05);
    }
    .sentence-card-hallucinated {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.08);
    }
    .sentence-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-supported {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-hallucinated {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-errtype {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        margin-left: 6px;
    }
    .score-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #cbd5e1;
    }

    /* Comparison Split Columns */
    .compare-col {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 16px;
        height: 100%;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loading & Caching Helpers
# -----------------------------------------------------------------------------
@st.cache_data
def load_all_dataset_data():
    """Load testset records, sample pairs, and predictions from disk."""
    testset_path = os.path.join(BASE_DIR, "data", "testset", "testset.csv")
    pairs_path = os.path.join(BASE_DIR, "data", "processed", "sample_pairs.jsonl")
    pred_a_path = os.path.join(BASE_DIR, "results", "approachA_mnli_base_predictions.csv")
    pred_b_path = os.path.join(BASE_DIR, "results", "approachB_selfcheck_predictions.csv")
    pred_h_path = os.path.join(BASE_DIR, "results", "approach_hybrid_predictions.csv")
    comp_summary_path = os.path.join(BASE_DIR, "results", "comparison_summary.csv")

    testset_df = pd.read_csv(testset_path) if os.path.exists(testset_path) else pd.DataFrame()
    pred_a = pd.read_csv(pred_a_path) if os.path.exists(pred_a_path) else pd.DataFrame()
    pred_b = pd.read_csv(pred_b_path) if os.path.exists(pred_b_path) else pd.DataFrame()
    pred_h = pd.read_csv(pred_h_path) if os.path.exists(pred_h_path) else pd.DataFrame()
    comp_df = pd.read_csv(comp_summary_path) if os.path.exists(comp_summary_path) else pd.DataFrame()

    # Load full judgment texts from sample_pairs.jsonl
    judgments_map = {}
    if os.path.exists(pairs_path):
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pair_id = rec.get("id")
                if pair_id:
                    judgments_map[pair_id] = rec

    # Merge predictions into testset_df
    merged = testset_df.copy()
    if not pred_a.empty:
        merged = merged.merge(pred_a[["row_id", "hallucination_score"]].rename(columns={"hallucination_score": "score_a"}), on="row_id", how="left")
    if not pred_b.empty:
        merged = merged.merge(pred_b[["row_id", "hallucination_score"]].rename(columns={"hallucination_score": "score_b"}), on="row_id", how="left")
    if not pred_h.empty:
        merged = merged.merge(pred_h[["row_id", "hallucination_score"]].rename(columns={"hallucination_score": "score_hybrid"}), on="row_id", how="left")

    return {
        "testset_df": testset_df,
        "merged_df": merged,
        "judgments_map": judgments_map,
        "comp_df": comp_df
    }


DATA = load_all_dataset_data()
merged_df = DATA["merged_df"]
judgments_map = DATA["judgments_map"]
comp_df = DATA["comp_df"]


# -----------------------------------------------------------------------------
# Top Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <span class="hero-badge">Deep Learning & NLP Research • Sem 5 Mini-Project</span>
        <h1 class="hero-title">⚖️ Legal Hallucination Detector</h1>
        <p class="hero-subtitle">
            Benchmarking <strong>Cross-Encoder NLI (SummaC)</strong> vs. 
            <strong>Black-Box Self-Consistency (SelfCheckGPT)</strong> on 
            <strong>Indian Court Judgments (MILDSum Benchmark)</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# Sidebar Controls & Global Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Evaluation Controls")
    
    # 1. Preset Case Selector
    case_ids = [f"Sample_{i}" for i in range(1, 11)]
    selected_case = st.selectbox(
        "Select Legal Case (MILDSum):",
        options=case_ids,
        index=0,
        help="Choose one of 10 Indian court judgment cases from the evaluated test set."
    )

    # 2. Model Selection
    model_choice = st.selectbox(
        "Hallucination Detection Engine:",
        options=[
            "Approach A: SummaC NLI (DeBERTa-v3-MNLI)",
            "Approach B: SelfCheckGPT (Consistency)",
            "Hybrid Ensemble (Calibrated Fusion)"
        ],
        index=0
    )

    # 3. Dynamic Threshold Slider
    st.markdown("---")
    st.markdown("#### 🎯 Decision Threshold ($\theta$)")
    
    if "Approach A" in model_choice:
        default_theta = 0.85
        theta_help = "Domain-calibrated threshold for SummaC NLI. At θ=0.85, precision improves while retaining 100% recall."
    elif "Approach B" in model_choice:
        default_theta = 0.50
        theta_help = "SelfCheckGPT threshold: sentences with consistency < 0.50 are flagged as hallucinations."
    else:
        default_theta = 0.50
        theta_help = "Calibrated ensemble threshold combining NLI logical verification and stochastic consistency."

    threshold = st.slider(
        "Threshold (Higher = Stricter flag):",
        min_value=0.10,
        max_value=0.95,
        value=default_theta,
        step=0.01,
        help=theta_help
    )

    # Threshold reset button
    if st.button("Reset to Recommended θ", use_container_width=True):
        st.session_state["threshold_override"] = default_theta
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📌 Benchmark Highlights")
    st.info(
        "• **Approach A:** 100% Recall (16/16 detected), ROC-AUC = 0.8585\n\n"
        "• **Approach B:** 92.4% Accuracy, 100% on pure fabrications, 0.51s latency\n\n"
        "• **Testset:** 462 sentences (16 ground-truth legal mutations)"
    )


# -----------------------------------------------------------------------------
# Main Content Tabs
# -----------------------------------------------------------------------------
tab_inspect, tab_compare, tab_dashboard, tab_playground, tab_methodology = st.tabs([
    "🔍 Sentence Inspector",
    "⚖️ Side-by-Side Comparison",
    "📊 Benchmark Dashboard",
    "🧪 Live Custom Playground",
    "📖 Methodology & Viva Guide"
])


# =============================================================================
# TAB 1: SENTENCE-LEVEL HALLUCINATION INSPECTOR
# =============================================================================
with tab_inspect:
    # Filter sentences for selected case
    case_sentences = merged_df[merged_df["pair_id"] == selected_case].copy()
    case_judgment = judgments_map.get(selected_case, {})
    judgment_text = case_judgment.get("judgment_text", "Source judgment text not loaded.")
    
    # Extract model scores for selected model
    if "Approach A" in model_choice:
        scores = case_sentences["score_a"].values
        model_name_tag = "SummaC Cross-Encoder NLI"
    elif "Approach B" in model_choice:
        scores = case_sentences["score_b"].values
        model_name_tag = "SelfCheckGPT Consistency"
    else:
        scores = case_sentences["score_hybrid"].values
        model_name_tag = "Hybrid Ensemble"

    # Compute case-level stats at current threshold
    pred_labels = (scores >= threshold).astype(int)
    true_labels = case_sentences["label"].values
    
    n_total = len(case_sentences)
    n_flagged = int(np.sum(pred_labels))
    n_true_pos = int(np.sum((pred_labels == 1) & (true_labels == 1)))
    n_false_pos = int(np.sum((pred_labels == 1) & (true_labels == 0)))
    n_ground_truth_pos = int(np.sum(true_labels == 1))

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Summary Sentences</div>
                <div class="metric-value">{n_total}</div>
                <div class="metric-caption">In case {selected_case}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Ground Truth Hallucinations</div>
                <div class="metric-value" style="color: {'#f87171' if n_ground_truth_pos > 0 else '#34d399'};">{n_ground_truth_pos}</div>
                <div class="metric-caption">Injected legal errors</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Flagged by Model</div>
                <div class="metric-value">{n_flagged}</div>
                <div class="metric-caption">At θ = {threshold:.2f}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c4:
        detection_rate = f"{(n_true_pos / max(n_ground_truth_pos, 1) * 100):.0f}%" if n_ground_truth_pos > 0 else "N/A"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Case Recall</div>
                <div class="metric-value" style="color: #38bdf8;">{detection_rate}</div>
                <div class="metric-caption">{n_true_pos} of {n_ground_truth_pos} detected</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main split layout: Judgment Source vs Summary Sentence Cards
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📜 Source Court Judgment")
        st.caption(f"Indian Supreme Court judgment text for **{selected_case}** ({len(judgment_text.split()):,} words)")
        with st.expander("Click to view full judgment text", expanded=False):
            st.text_area("Judgment Text", value=judgment_text, height=450, disabled=True)

    with col_right:
        st.markdown("### 📝 AI Summary Sentence Analysis")
        st.caption(f"Inspected by **{model_name_tag}** at threshold **θ = {threshold:.2f}**")

        for idx, (_, row) in enumerate(case_sentences.iterrows()):
            score = float(scores[idx]) if not np.isnan(scores[idx]) else 0.0
            is_flagged = score >= threshold
            is_ground_truth_hallucinated = int(row["label"]) == 1
            error_type = str(row.get("error_type", "")) if pd.notna(row.get("error_type")) else ""
            orig_sentence = str(row.get("original_sentence", "")) if pd.notna(row.get("original_sentence")) else ""

            card_class = "sentence-card-hallucinated" if is_flagged else "sentence-card-supported"
            badge_html = f'<span class="badge-pill badge-hallucinated">⚠️ Hallucination</span>' if is_flagged else f'<span class="badge-pill badge-supported">✓ Supported</span>'
            err_html = f'<span class="badge-pill badge-errtype">{error_type}</span>' if error_type else ""

            st.markdown(
                f"""
                <div class="sentence-card {card_class}">
                    <div class="sentence-meta">
                        <div>
                            <strong>Sentence #{idx + 1}</strong> &nbsp;
                            {badge_html} {err_html}
                        </div>
                        <div class="score-badge">Score: <strong>{score:.4f}</strong></div>
                    </div>
                    <div style="font-size: 0.95rem; color: #f1f5f9; line-height: 1.5; margin-top: 6px;">
                        {row['sentence_text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # If hallucinated or error details exist, show an expandable inspection drawer
            if is_ground_truth_hallucinated or is_flagged:
                with st.expander(f"Inspector Details: Sentence #{idx + 1}"):
                    st.write(f"**Hallucination Probability:** `{score:.4f}` (Threshold: `{threshold:.2f}`)")
                    st.write(f"**Prediction Status:** {'🔴 Positive (Flagged)' if is_flagged else '🟢 Negative (Passed)'}")
                    st.write(f"**Ground Truth Label:** {'Hallucinated (1)' if is_ground_truth_hallucinated else 'Supported (0)'}")
                    if is_ground_truth_hallucinated:
                        st.markdown(f"**Injected Error Type:** `{error_type}`")
                        st.markdown(f"**Original Factual Sentence:** *\"{orig_sentence}\"*")
                        st.markdown(f"**Mutated Sentence:** *\"{row['sentence_text']}\"*")


# =============================================================================
# TAB 2: SIDE-BY-SIDE MODEL COMPARISON
# =============================================================================
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Model Comparison")
    st.caption(f"Compare **Approach A (NLI)** vs. **Approach B (SelfCheckGPT)** on **{selected_case}**")

    case_sentences = merged_df[merged_df["pair_id"] == selected_case].copy()
    scores_a = case_sentences["score_a"].values
    scores_b = case_sentences["score_b"].values
    true_labels = case_sentences["label"].values

    c_comp1, c_comp2 = st.columns(2)

    with c_comp1:
        st.markdown(
            """
            <div class="compare-col">
                <h4 style="color: #38bdf8; margin-top: 0;">Approach A: SummaC NLI</h4>
                <p style="font-size: 0.85rem; color: #94a3b8;">
                    Cross-Encoder NLI (DeBERTa-v3-MNLI) verifying logical entailment against source judgment.
                    <strong>Operating Threshold: θ = 0.85</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        for idx, (_, row) in enumerate(case_sentences.iterrows()):
            score_a = float(scores_a[idx]) if not np.isnan(scores_a[idx]) else 0.0
            is_flagged = score_a >= 0.85
            badge_html = f'<span class="badge-pill badge-hallucinated">⚠️ Flagged</span>' if is_flagged else f'<span class="badge-pill badge-supported">✓ Supported</span>'
            
            st.markdown(
                f"""
                <div class="sentence-card {'sentence-card-hallucinated' if is_flagged else 'sentence-card-supported'}" style="padding: 10px 14px;">
                    <div class="sentence-meta">
                        <span><strong>#{idx+1}</strong> {badge_html}</span>
                        <span class="score-badge">H: {score_a:.3f}</span>
                    </div>
                    <div style="font-size: 0.88rem; color: #e2e8f0;">{row['sentence_text']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with c_comp2:
        st.markdown(
            """
            <div class="compare-col">
                <h4 style="color: #34d399; margin-top: 0;">Approach B: SelfCheckGPT</h4>
                <p style="font-size: 0.85rem; color: #94a3b8;">
                    Black-box self-consistency checking against stochastic sampled summaries (N=3).
                    <strong>Operating Threshold: θ = 0.50</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        for idx, (_, row) in enumerate(case_sentences.iterrows()):
            score_b = float(scores_b[idx]) if not np.isnan(scores_b[idx]) else 0.0
            is_flagged = score_b >= 0.50
            badge_html = f'<span class="badge-pill badge-hallucinated">⚠️ Flagged</span>' if is_flagged else f'<span class="badge-pill badge-supported">✓ Supported</span>'
            
            st.markdown(
                f"""
                <div class="sentence-card {'sentence-card-hallucinated' if is_flagged else 'sentence-card-supported'}" style="padding: 10px 14px;">
                    <div class="sentence-meta">
                        <span><strong>#{idx+1}</strong> {badge_html}</span>
                        <span class="score-badge">H: {score_b:.3f}</span>
                    </div>
                    <div style="font-size: 0.88rem; color: #e2e8f0;">{row['sentence_text']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =============================================================================
# TAB 3: BENCHMARK PERFORMANCE DASHBOARD
# =============================================================================
with tab_dashboard:
    st.markdown("### 📊 Benchmark Performance Dashboard")
    st.caption("Standardized evaluation across 462 Indian court judgment sentences (`data/testset/testset.csv`)")

    if not comp_df.empty:
        # Format metrics table for display
        display_df = comp_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg Precision"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Comparison summary table not found. Run scripts/compare_approaches.py to generate it.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Dashboard Charts
    chart_col1, chart_col2 = st.columns(2)
    
    chart_path = os.path.join(BASE_DIR, "results", "comparison_breakdown.png")
    roc_path = os.path.join(BASE_DIR, "results", "roc_curve_approach_a.png")

    with chart_col1:
        st.markdown("#### 📈 Multi-Approach Comparative Evaluation")
        if os.path.exists(chart_path):
            st.image(chart_path, caption="Figure 1: Core Metrics, Latency Trade-Off, Error Breakdown & Confusion Matrix", use_container_width=True)
        else:
            st.info("Chart not generated. Run python scripts/compare_approaches.py")

    with chart_col2:
        st.markdown("#### 🎯 Approach A ROC-AUC Curve")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Figure 2: Receiver Operating Characteristic (ROC-AUC = 0.8585) for SummaC NLI", use_container_width=True)
        else:
            st.info("ROC Curve not found. Run python scripts/run_approach_a.py")


# =============================================================================
# TAB 4: LIVE CUSTOM PLAYGROUND
# =============================================================================
with tab_playground:
    st.markdown("### 🧪 Live Hallucination Playground")
    st.caption("Test custom Indian court judgments and candidate summaries in real time.")

    demo_judgment = (
        "The appellant, Smt. Rekha Sharma, filed the present criminal appeal against the judgment of the High Court of Delhi "
        "dated 14th August 2018. The High Court had upheld the conviction of the appellant under Section 302 read with Section 34 "
        "of the Indian Penal Code, sentencing her to life imprisonment and a fine of Rs. 25,000. The prosecution alleged that the "
        "incident occurred on 5th January 2015 at Model Town, New Delhi. After examining the testimony of PW-1 and ballistic forensic "
        "reports, this Court finds significant material contradictions in the prosecution story. Consequently, the conviction cannot "
        "be sustained. The appeal is allowed and the appellant is acquitted of all charges. The appellant shall be released forthwith "
        "unless required in any other case."
    )

    demo_summary = (
        "The Supreme Court heard an appeal by Smt. Rekha Sharma challenging her conviction under Section 302 IPC.\n"
        "The High Court of Delhi had previously sentenced her to life imprisonment with a fine of Rs. 25,000.\n"
        "The Supreme Court dismissed the appeal and confirmed the death sentence of the appellant.\n"
        "The Court also directed the state to pay compensation of Rs. 50,00,000 to the victim's family."
    )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        custom_judgment = st.text_area("Judgment Source Text:", value=demo_judgment, height=220)
    with p_col2:
        custom_summary = st.text_area("AI-Generated Summary Text:", value=demo_summary, height=220)

    if st.button("🚀 Analyze Summary for Hallucinations", type="primary"):
        with st.spinner("Analyzing sentences against judgment..."):
            cleaned_summary = clean_text(custom_summary)
            sents = split_sentences(cleaned_summary)
            
            st.markdown(f"#### Analysis Results ({len(sents)} sentences detected):")
            
            # Fast containment checking against source judgment
            from hd.approach_b import compute_token_containment
            
            for i, sent in enumerate(sents):
                containment = compute_token_containment(sent, custom_judgment)
                hallucination_score = round(1.0 - containment, 4)
                is_hallucinated = hallucination_score >= 0.50
                
                badge_html = f'<span class="badge-pill badge-hallucinated">⚠️ Hallucination</span>' if is_hallucinated else f'<span class="badge-pill badge-supported">✓ Supported</span>'
                card_class = "sentence-card-hallucinated" if is_hallucinated else "sentence-card-supported"

                st.markdown(
                    f"""
                    <div class="sentence-card {card_class}">
                        <div class="sentence-meta">
                            <span><strong>Sentence #{i+1}</strong> {badge_html}</span>
                            <span class="score-badge">Consistency Score: <strong>{containment*100:.1f}%</strong> | Hallucination Score: <strong>{hallucination_score:.4f}</strong></span>
                        </div>
                        <div style="font-size: 0.95rem; color: #f8fafc; margin-top: 6px;">{sent}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =============================================================================
# TAB 5: METHODOLOGY & VIVA GUIDE
# =============================================================================
with tab_methodology:
    st.markdown("### 📖 Scientific Methodology & Viva Preparation Guide")
    
    st.markdown(
        """
        #### 1. Core Research Problem
        When Large Language Models (LLMs) summarize complex legal documents like Indian court judgments, 
        they frequently hallucinate **counter-factual details**—swapping case outcomes (e.g. *granted* vs *dismissed*), 
        modifying monetary awards, transposing dates, or inventing procedural directives that never occurred.
        
        #### 2. The Two Detection Paradigms
        
        | Feature | Approach A: SummaC NLI | Approach B: SelfCheckGPT |
        | :--- | :--- | :--- |
        | **Paradigm** | White-Box Natural Language Inference (NLI) | Black-Box Stochastic Self-Consistency |
        | **Backbone Model** | `microsoft/deberta-base-mnli` | Any LLM (Llama 3 / Mistral / DeepSeek via Ollama) |
        | **Input Required** | Candidate sentence + Full Court Judgment | Candidate sentence + $N=3$ stochastic summaries |
        | **Scoring Formula** | $H(s) = 1.0 - \max_i P(\text{Entailment} \mid C_i, s)$ | $H(s) = 1.0 - \frac{1}{K} \sum_{k=1}^K \text{Containment}(s, S_k)$ |
        | **Key Strength** | **100% Recall:** Catches subtle single-word date/negation/outcome flips | **92.4% Accuracy:** Almost zero false alarms, instantaneous latency |
        | **Limitation** | False positives on dense legal prose at default $\theta=0.5$ | Misses single-word antonym flips where lexical overlap is high |
        | **Inference Latency** | ~107 seconds on T4 GPU (evaluates ~35,000 sentence pairs) | **0.51 seconds on CPU** |

        #### 3. Key Viva Questions & Model Answers
        
        **Q1: Why did Approach A need a domain-calibrated threshold ($\theta = 0.85$)?**  
        *Answer:* The MNLI dataset used to pre-train DeBERTa consists of simple, short conversational sentences. 
        In contrast, Indian legal judgments contain complex, multi-clause sentences with legal terms of art. 
        Because MNLI models penalize stylistic mismatches as "neutral" rather than "entailment", uncalibrated 
        support scores cluster around 0.65–0.75 even for supported sentences. Adjusting $\theta$ from 0.50 to 0.85 
        aligns the threshold with legal language density, raising overall accuracy from 27.3% to **72.3%** while 
        preserving **100% recall**.

        **Q2: Why can Approach B catch pure fabrications with 100% recall, but miss date mutations?**  
        *Answer:* Fabricated sentences represent orders or parties that do not exist in the judgment. When the LLM 
        samples independent summaries, this hallucinated order does not reappear, resulting in 0% containment and 
        a high hallucination score. Conversely, when a date is mutated (e.g. *2015* to *2021*), 90%+ of the sentence 
        tokens remain identical; pure unigram containment retains high overlap, meaning black-box lexical consistency 
        cannot detect the single-token contradiction without an NLI classifier.

        **Q3: What is the optimal pipeline for a real-world legal AI application?**  
        *Answer:* A **two-tiered hybrid pipeline**:
        1. Run **Approach B (SelfCheckGPT)** first on CPU (~0.5s) to immediately flag large-scale hallucinations and fabrications.
        2. Run **Approach A (Cross-Encoder NLI)** on the top candidate sentences to verify fine-grained numerical, temporal, and legal outcome consistency.
        """
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem;">
        Legal Hallucination Detector • B.Tech Sem 5 Mini-Project (Deep Learning + NLP) • 
        Built with Streamlit, PyTorch, Transformers, SummaC & SelfCheckGPT
    </div>
    """,
    unsafe_allow_html=True
)
