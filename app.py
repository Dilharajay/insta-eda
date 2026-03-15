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
st.caption("Drop in a CSV. Get a full EDA report + AI-recommended visuals — powered by Gemini.")
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
    st.markdown("**AI Visuals:** Smart Chart Selection")
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
            if "viz_configs" in st.session_state:
                del st.session_state["viz_configs"]

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
                            viz_configs = result_data["viz_configs"]

                            report_md = inject_metadata(
                                report_md,
                                dataset_name=uploaded_file.name,
                                rows=df.shape[0],
                                cols=df.shape[1],
                            )

                            st.session_state["report"] = report_md
                            st.session_state["raw_results"] = raw_results
                            st.session_state["viz_configs"] = viz_configs
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
        
        viz_tab, report_tab = st.tabs(["📊 AI-Recommended Visuals", "📄 Markdown Report"])

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
            st.subheader("📊 Data Exploration Dashboard (AI-Selected)")
            raw = st.session_state["raw_results"]
            viz_configs = st.session_state.get("viz_configs", [])
            
            if not viz_configs:
                st.info("Gemini didn't recommend specific visuals for this dataset. Here is the default analysis.")
                # Fallback to default visuals
                viz_configs = [
                    {"tool": "missing_values", "chart_type": "bar", "title": "Missing Values Overview"},
                    {"tool": "outlier_detection", "chart_type": "bar", "title": "Outlier Counts"},
                    {"tool": "correlation_analysis", "chart_type": "heatmap", "title": "Top Correlations"},
                    {"tool": "categorical_analysis", "chart_type": "pie", "title": "Categorical Distributions"}
                ]

            for config in viz_configs:
                tool_key = config.get("tool")
                chart_type = config.get("chart_type")
                title = config.get("title", "Insight Chart")
                description = config.get("description", "")
                
                try:
                    tool_data = json.loads(raw[tool_key]) if tool_key in raw and isinstance(raw[tool_key], str) else raw.get(tool_key)
                    if not tool_data or (isinstance(tool_data, str) and tool_data.startswith("No")):
                        continue

                    st.write(f"### {title}")
                    if description:
                        st.caption(description)

                    if tool_key == "missing_values":
                        mv_df = pd.DataFrame.from_dict(tool_data, orient='index').reset_index()
                        mv_df.columns = ['Column', 'Missing Count', 'Percent', 'Concern']
                        fig = px.bar(mv_df, x='Column', y='Percent', color='Concern', 
                                   title=title, color_discrete_map={True: '#ef553b', False: '#636efa'})
                        st.plotly_chart(fig, use_container_width=True)

                    elif tool_key == "outlier_detection":
                        out_df = pd.DataFrame.from_dict(tool_data, orient='index').reset_index()
                        out_df.columns = ['Column', 'Outlier Count', 'Percent', 'Lower', 'Upper']
                        fig = px.bar(out_df, x='Column', y='Outlier Count', 
                                    title=title, color_discrete_sequence=['#ab63fa'])
                        st.plotly_chart(fig, use_container_width=True)

                    elif tool_key == "correlation_analysis":
                        corr_df = pd.DataFrame(tool_data)
                        if chart_type == "heatmap" or chart_type == "bar":
                            fig = px.bar(corr_df, x='correlation', y='feature_a', color='feature_b',
                                         orientation='h', title=title)
                            st.plotly_chart(fig, use_container_width=True)

                    elif tool_key == "categorical_analysis":
                        # For categorical, we might have multiple columns. AI might specify which one in params, 
                        # but for now let's show all or first 3
                        cols = st.columns(min(len(tool_data), 3))
                        for i, (col_name, info) in enumerate(tool_data.items()):
                            if i >= 3: break 
                            with cols[i]:
                                top_v = pd.DataFrame.from_dict(info['top_5_values'], orient='index').reset_index()
                                top_v.columns = ['Value', 'Count']
                                fig = px.pie(top_v, names='Value', values='Count', title=f"{col_name}")
                                st.plotly_chart(fig, use_container_width=True)

                    elif tool_key == "descriptive_stats":
                        # Descriptive stats are hard to visualize as a single chart without more context,
                        # but we can show a bar chart of means or std devs if AI asks.
                        stats_df = pd.DataFrame(tool_data).T.reset_index()
                        stats_df.columns = ['Feature', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                        fig = px.bar(stats_df, x='Feature', y='mean', error_y='std', title=title)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    # Silently skip errors for individual charts
                    pass

    elif not uploaded_file:
        st.info("Upload a CSV file above to get started.")
