"""
InstaEDA — Streamlit UI
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import io
import json
import plotly.express as px

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
st.caption("Drop in a CSV. Get a full EDA report + interactive visuals instantly — powered by Gemini.")
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
    st.caption("Built with LangChain + Gemini + Plotly")

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
    st.markdown("**Visuals:** Plotly Interactive Charts")
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
            if "raw_results" in st.session_state:
                del st.session_state["raw_results"]

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
                            result_data = run_eda(df, api_key=api_key, model_name=model_name)
                            report_md = result_data["report"]
                            raw_results = result_data["raw_results"]

                            report_md = inject_metadata(
                                report_md,
                                dataset_name=uploaded_file.name,
                                rows=df.shape[0],
                                cols=df.shape[1],
                            )

                            st.session_state["report"] = report_md
                            st.session_state["raw_results"] = raw_results
                            st.session_state["dataset_name"] = uploaded_file.name

                        except Exception as e:
                            st.error(f"Agent error: {e}")

        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    # ─────────────────────────────────────────────
    # Report & Visuals Display
    # ─────────────────────────────────────────────
    if "report" in st.session_state and "raw_results" in st.session_state:
        st.divider()
        
        viz_tab, report_tab = st.tabs(["📊 Interactive Visuals", "📄 Markdown Report"])

        with report_tab:
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

        with viz_tab:
            st.subheader("📊 Data Exploration Dashboard")
            raw = st.session_state["raw_results"]
            
            # 1. Missing Values Chart
            try:
                mv_data = json.loads(raw["missing_values"])
                if isinstance(mv_data, dict):
                    mv_df = pd.DataFrame.from_dict(mv_data, orient='index').reset_index()
                    mv_df.columns = ['Column', 'Missing Count', 'Percent', 'Concern']
                    fig_mv = px.bar(mv_df, x='Column', y='Percent', color='Concern', 
                                   title="Missing Values (%) by Column",
                                   color_discrete_map={True: '#ef553b', False: '#636efa'})
                    st.plotly_chart(fig_mv, use_container_width=True)
                else:
                    st.info("No missing values detected.")
            except: pass

            # 2. Outlier Analysis
            try:
                out_data = json.loads(raw["outlier_detection"])
                if isinstance(out_data, dict):
                    out_df = pd.DataFrame.from_dict(out_data, orient='index').reset_index()
                    out_df.columns = ['Column', 'Outlier Count', 'Percent', 'Lower', 'Upper']
                    fig_out = px.bar(out_df, x='Column', y='Outlier Count', 
                                    title="Outlier Count by Column",
                                    color_discrete_sequence=['#ab63fa'])
                    st.plotly_chart(fig_out, use_container_width=True)
                else:
                    st.info("No outliers detected.")
            except: pass

            # 3. Categorical Distribution (Top 1)
            try:
                cat_data = json.loads(raw["categorical_analysis"])
                if isinstance(cat_data, dict):
                    st.write("#### Categorical Distributions (Top 5 Values)")
                    cols = st.columns(min(len(cat_data), 3))
                    for i, (col_name, info) in enumerate(cat_data.items()):
                        with cols[i % 3]:
                            top_v = pd.DataFrame.from_dict(info['top_5_values'], orient='index').reset_index()
                            top_v.columns = ['Value', 'Count']
                            fig_cat = px.pie(top_v, names='Value', values='Count', title=f"{col_name}")
                            st.plotly_chart(fig_cat, use_container_width=True)
            except: pass

            # 4. Correlation Heatmap (Top pairs)
            try:
                corr_data = json.loads(raw["correlation_analysis"])
                if isinstance(corr_data, list):
                    corr_df = pd.DataFrame(corr_data)
                    fig_corr = px.bar(corr_df, x='correlation', y='feature_a', color='feature_b',
                                     orientation='h', title="Top Feature Correlations",
                                     labels={'feature_a': 'Feature A', 'correlation': 'Correlation Coefficient'})
                    st.plotly_chart(fig_corr, use_container_width=True)
            except: pass

    elif not uploaded_file:
        st.info("Upload a CSV file above to get started.")
