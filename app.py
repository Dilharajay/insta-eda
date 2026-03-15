"""
InstaEDA — Streamlit UI
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
# Sidebar — Branding & Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚡ InstaEDA v1.0")
    st.markdown("""
InstaEDA automatically analyzes:
- **Dataset shape** (rows, cols, types)
- **Missing values** (null analysis)
- **Statistics** (numeric summary)
- **Outliers** (IQR-based detection)
- **Correlations** (feature pairs)
- **Categorical** (cardinality check)
- **ML Advice** (problem type suggestion)
""")
    st.divider()
    st.caption("Built with LangChain + Gemini")

# ─────────────────────────────────────────────
# Tabs — Run and Settings
# ─────────────────────────────────────────────
tab_run, tab_settings = st.tabs(["🚀 Analysis", "⚙️ Settings"])

with tab_settings:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza...",
        help="Your key is never stored or logged. Get one at aistudio.google.com",
    )
    
    model_name = st.selectbox(
        "Gemini Model Selection",
        options=[
            "gemini-3.1-pro",
            "gemini-3.1-flash-lite",
            "gemini-3-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        index=4,  # default to gemini-2.5-flash
        help="Choose the model that powers the analysis. Pro models are more capable, Flash models are faster.",
    )
    
    st.divider()
    st.markdown(f"**Current Model:** `{model_name}`")
    st.markdown("**Tools:** 7 automated EDA tools")
    st.markdown("**Output:** Markdown report with ML recommendations")

with tab_run:
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"],
        help="Accepts any standard CSV file.",
    )

    if uploaded_file:
        # Clear previous report if a new file is uploaded
        if "last_file" not in st.session_state or st.session_state["last_file"] != uploaded_file.name:
            st.session_state["last_file"] = uploaded_file.name
            if "report" in st.session_state:
                del st.session_state["report"]

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
                    st.error("Please enter your Google API key in the **Settings** tab.")
                else:
                    with st.spinner("Agent is analyzing your dataset... this may take 30–60 seconds."):
                        try:
                            # Pass api_key and model_name to the agent
                            report_md = run_eda(df, api_key=api_key, model_name=model_name)
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

    # Report Display
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

    elif not uploaded_file:
        st.info("Upload a CSV file above to get started.")
