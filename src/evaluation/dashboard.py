"""Streamlit evaluation dashboard — RAGAS metrics over time.

Run:
    streamlit run src/evaluation/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings
from src.evaluation.metrics_store import MetricsStore

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CERN Knowledge Navigator — Evaluation Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0d1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .pass { color: #3fb950; font-weight: bold; }
    .fail { color: #f85149; font-weight: bold; }
    h1 { color: #58a6ff; }
    h3 { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚛️ CERN Knowledge Navigator")
st.subheader("RAGAS Evaluation Dashboard")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(f"**LLM**: `{settings.groq_model}`")
    st.markdown(f"**Judge**: `{settings.groq_judge_model}`")
    st.markdown(f"**Embeddings**: `{settings.embedding_model}`")
    st.markdown(f"**Faithfulness Threshold**: `{settings.faithfulness_threshold}`")
    st.markdown("---")
    st.markdown("**Run new evaluation:**")
    st.code("python -m src.evaluation.ragas_runner --sample 10", language="bash")

# ── Load data ─────────────────────────────────────────────────────────────────
store = MetricsStore()
history = store.get_history(limit=100)
latest = store.get_latest()

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
THRESHOLD = settings.faithfulness_threshold

if not history:
    st.warning(
        "No evaluation runs found. Run the evaluation pipeline first:\n\n"
        "```bash\npython -m src.evaluation.ragas_runner --sample 10\n```"
    )
    st.stop()

df = pd.DataFrame(history)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── Latest run KPIs ───────────────────────────────────────────────────────────
st.header("📊 Latest Evaluation Run")

if latest:
    cols = st.columns(4)
    metric_labels = {
        "faithfulness": ("🎯 Faithfulness", THRESHOLD),
        "answer_relevancy": ("💬 Answer Relevancy", 0.7),
        "context_precision": ("🔍 Context Precision", 0.7),
        "context_recall": ("📚 Context Recall", 0.7),
    }
    for col, (key, (label, thr)) in zip(cols, metric_labels.items()):
        val = latest.get(key)
        if val is not None:
            delta_color = "normal" if val >= thr else "inverse"
            col.metric(
                label=label,
                value=f"{val:.3f}",
                delta=f"{'✓ PASS' if val >= thr else '✗ FAIL'} (thr={thr})",
                delta_color=delta_color,
            )
        else:
            col.metric(label=label, value="N/A")

    st.caption(
        f"Run ID: `{latest.get('run_id')}` | "
        f"Timestamp: `{latest.get('timestamp', '')[:19]}` | "
        f"Questions: `{latest.get('question_count', 'N/A')}`"
    )

st.markdown("---")

# ── Radar chart ───────────────────────────────────────────────────────────────
st.header("🕸️ Metrics Radar")

if latest:
    radar_metrics = [m for m in METRIC_COLS if latest.get(m) is not None]
    radar_vals = [latest[m] for m in radar_metrics]
    radar_labels = [m.replace("_", " ").title() for m in radar_metrics]

    fig_radar = go.Figure(
        go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(88, 166, 255, 0.2)",
            line=dict(color="#58a6ff", width=2),
            name="Latest Run",
        )
    )
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        height=400,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── Time series ───────────────────────────────────────────────────────────────
st.header("📈 Metrics Over Time")

metric_colors = {
    "faithfulness": "#58a6ff",
    "answer_relevancy": "#3fb950",
    "context_precision": "#f78166",
    "context_recall": "#d2a8ff",
}

fig_ts = go.Figure()
for metric, color in metric_colors.items():
    valid = df[df[metric].notna()]
    if not valid.empty:
        fig_ts.add_trace(
            go.Scatter(
                x=valid["timestamp"],
                y=valid[metric],
                mode="lines+markers",
                name=metric.replace("_", " ").title(),
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )

# Threshold line
fig_ts.add_hline(
    y=THRESHOLD,
    line_dash="dash",
    line_color="#f85149",
    annotation_text=f"Faithfulness Threshold ({THRESHOLD})",
    annotation_position="bottom right",
)

fig_ts.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9"),
    xaxis=dict(title="Evaluation Timestamp", gridcolor="#30363d"),
    yaxis=dict(title="Score", range=[0, 1], gridcolor="#30363d"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    height=400,
)
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# ── History table ─────────────────────────────────────────────────────────────
st.header("📋 Evaluation History")

display_df = df.copy()
display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
display_df["passes"] = display_df["faithfulness"] >= THRESHOLD
display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

st.dataframe(
    display_df.tail(20),
    use_container_width=True,
    hide_index=True,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:12px;'>"
    "CERN Knowledge Navigator | RAG + MCP + RAGAS | Built with LangChain + FastMCP + Groq"
    "</div>",
    unsafe_allow_html=True,
)
