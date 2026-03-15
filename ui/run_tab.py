import streamlit as st
import pandas as pd
from agent.eda_agent import run_eda
from utils.report import inject_metadata
from utils.auth import save_report_to_db

def render_run_tab():
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"],
        help="Accepts any standard CSV file.",
    )

    if uploaded_file:
        # Clear current report if a NEW file is uploaded manually
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
                api_key_input = st.session_state.get("api_key_input", "")
                model_name = st.session_state.get("model_name", "gemini-2.5-flash")
                
                if not api_key_input:
                    st.error("Please enter (and save) your Google API key in the **Settings** tab.")
                else:
                    with st.spinner("Agent is analyzing your dataset... this may take 30–60 seconds."):
                        try:
                            # Pass api_key and model_name to the agent
                            result_data = run_eda(df, api_key=api_key_input, model_name=model_name)
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
                            
                            # SAVE TO DATABASE
                            save_report_to_db(
                                st.session_state["username"],
                                uploaded_file.name,
                                report_md,
                                raw_results,
                                viz_configs
                            )
                            st.success("Analysis complete and saved to history!")

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
                    file_name=f"eda_{st.session_state.get('dataset_name', 'report').replace('.csv','')}.md",
                    mime="text/markdown",
                )
            st.markdown(st.session_state["report"])

        with viz_tab:
            from ui.visuals import render_visuals
            render_visuals()

    elif not uploaded_file:
        st.info("Upload a CSV file above to get started.")
