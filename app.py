"""
DataNarrator — Streamlit UI
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import io

from agent.eda_agent import run_eda
from utils.report import save_report, inject_metadata

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="InstaEDA",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("⚡ InstaEDA")
st.caption("Drop in a CSV. Get a full EDA report instantly — powered by Gemini.")
st.divider()

# ─────────────────────────────────────────────
# Sidebar — API Key + Settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza...",
        help="Your key is never stored or logged. Get one at aistudio.google.com",
    )
    st.markdown("---")
    st.markdown("**Model:** gemini-2.5-flash")
    st.markdown("**Tools used:** 7 EDA tools")
    st.markdown("**Output:** Markdown report")
    st.markdown("---")
    st.caption("InstaEDA v1.0 · Built with LangChain + Gemini")

# ─────────────────────────────────────────────
# File Upload
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help="Accepts any standard CSV file.",
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")

        # Preview
        with st.expander("👀 Preview dataset (first 5 rows)", expanded=False):
            st.dataframe(df.head(), use_container_width=True)

        st.divider()

        # Run button
        if st.button("🚀 Generate EDA Report", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your Google API key in the sidebar.")
            else:
                with st.spinner("Agent is analyzing your dataset... this may take 30–60 seconds."):
                    try:
                        report_md = run_eda(df, api_key=api_key)
                        report_md = inject_metadata(
                            report_md,
                            dataset_name=uploaded_file.name,
                            rows=df.shape[0],
                            cols=df.shape[1],
                        )

                        st.session_state["report"] = report_md
                        st.session_state["dataset_name"] = uploaded_file.name

                    except Exception as e:
                        st.error(f"Agent error: {e}")

    except Exception as e:
        st.error(f"Could not parse CSV: {e}")

# ─────────────────────────────────────────────
# Report Display
# ─────────────────────────────────────────────
if "report" in st.session_state:
    st.divider()
    st.subheader("📄 EDA Report")

    col1, col2 = st.columns([4, 1])
    with col2:
        st.download_button(
            label="⬇️ Download .md",
            data=st.session_state["report"],
            file_name=f"eda_{st.session_state['dataset_name'].replace('.csv','')}.md",
            mime="text/markdown",
        )

    st.markdown(st.session_state["report"])

# ─────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────
else:
    st.info("Upload a CSV file above to get started.")
    with st.expander("💡 What does DataNarrator analyze?"):
        st.markdown("""
- **Dataset shape** — rows, columns, dtypes
- **Missing values** — per-column null analysis with severity flags
- **Descriptive statistics** — mean, std, min, max, quartiles for all numeric features
- **Outlier detection** — IQR-based flagging per column
- **Correlations** — top feature pairs by Pearson correlation
- **Categorical features** — cardinality and value frequency analysis
- **ML recommendations** — problem type suggestion and starter sklearn pipeline
        """)