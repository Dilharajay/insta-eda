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
from utils.auth import (
    init_db, register_user, authenticate_user, 
    save_api_key, get_api_key, save_report_to_db, 
    get_user_reports, delete_report, update_username, 
    update_password
)

# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────
init_db()

# Page Config
st.set_page_config(
    page_title="InstaEDA",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────────
# User Authentication System
# ─────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

with st.sidebar:
    st.header("🔐 User Account")
    
    if not st.session_state["authenticated"]:
        mode = st.radio("Mode", ["Login", "Sign Up"])
        user_input = st.text_input("Username")
        pass_input = st.text_input("Password", type="password")
        
        if mode == "Login":
            if st.button("Login", type="primary", use_container_width=True):
                if authenticate_user(user_input, pass_input):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = user_input
                    st.success("Welcome back!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        else:
            if st.button("Register", type="primary", use_container_width=True):
                if register_user(user_input, pass_input):
                    st.success("Account created! You can now login.")
                else:
                    st.error("User already exists.")
    else:
        st.write(f"Logged in as: **{st.session_state['username']}**")
        if st.button("Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            st.rerun()

        st.divider()
        st.header("📜 Analysis History")
        reports = get_user_reports(st.session_state["username"])
        if reports:
            report_options = {f"{r[1]} ({r[5]})": r for r in reports}
            selected_report_label = st.selectbox("Select a past report", ["Select..."] + list(report_options.keys()))
            
            if selected_report_label != "Select...":
                report_data = report_options[selected_report_label]
                if st.button("📂 Load Report", use_container_width=True):
                    st.session_state["report"] = report_data[2]
                    st.session_state["raw_results"] = json.loads(report_data[3])
                    st.session_state["viz_configs"] = json.loads(report_data[4])
                    st.session_state["dataset_name"] = report_data[1]
                    st.success(f"Loaded {report_data[1]}")
                
                if st.button("🗑️ Delete Report", use_container_width=True):
                    delete_report(report_data[0])
                    st.success("Deleted report.")
                    st.rerun()
        else:
            st.caption("No past analyses found.")

    st.divider()
    st.header("⚡ InstaEDA v1.0")
    st.caption("Built with LangChain + Gemini + Plotly")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("⚡ InstaEDA")
st.caption("Drop in a CSV. Get a full EDA report + AI-recommended visuals — powered by Gemini.")
st.divider()

if not st.session_state["authenticated"]:
    st.info("Please login or register in the sidebar to start using InstaEDA.")
    st.stop()

# ─────────────────────────────────────────────
# Main Application Flow
# ─────────────────────────────────────────────

# Tabs — Run, Settings, and Account
tab_run, tab_settings, tab_account = st.tabs(["🚀 Analysis", "⚙️ API Settings", "👤 Account Management"])

with tab_account:
    st.header("👤 Account Management")
    
    with st.expander("Update Username"):
        st.info("Changing your username will update all your past reports as well.")
        new_un = st.text_input("New Username", key="new_un")
        if st.button("Change Username"):
            if not new_un:
                st.error("Username cannot be empty.")
            elif new_un == st.session_state["username"]:
                st.warning("New username is the same as the current one.")
            else:
                if update_username(st.session_state["username"], new_un):
                    st.session_state["username"] = new_un
                    st.success(f"Username successfully changed to **{new_un}**!")
                    st.rerun()
                else:
                    st.error("That username is already taken. Please choose another.")

    with st.expander("Update Password"):
        old_pw = st.text_input("Current Password", type="password", key="old_pw")
        new_pw = st.text_input("New Password", type="password", key="new_pw")
        conf_pw = st.text_input("Confirm New Password", type="password", key="conf_pw")
        
        if st.button("Change Password"):
            if not old_pw or not new_pw or not conf_pw:
                st.error("Please fill in all password fields.")
            elif new_pw != conf_pw:
                st.error("New passwords do not match.")
            elif new_pw == old_pw:
                st.warning("New password must be different from the old one.")
            else:
                # Verify old password
                if authenticate_user(st.session_state["username"], old_pw):
                    update_password(st.session_state["username"], new_pw)
                    st.success("Password updated successfully!")
                else:
                    st.error("Incorrect current password.")

with tab_settings:
    st.header("⚙️ Configuration")
    
    # Retrieve existing API key
    existing_key = get_api_key(st.session_state["username"])
    
    api_key_input = st.text_input(
        "Google API Key",
        value=existing_key if existing_key else "",
        type="password",
        placeholder="AIza...",
        help="Your key is stored securely in the database for your next session.",
    )
    
    if st.button("Save API Key"):
        save_api_key(st.session_state["username"], api_key_input)
        st.success("API key saved for your account!")
    
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
    st.markdown(f"**Account:** `{st.session_state['username']}`")
    st.markdown(f"**Current Model:** `{model_name}`")
    st.markdown("**Tools:** 7 automated EDA tools")
    st.markdown("**AI Visuals:** Smart Chart Selection")

with tab_run:
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
            st.subheader("📊 Data Exploration Dashboard (AI-Selected)")
            raw = st.session_state["raw_results"]
            viz_configs = st.session_state.get("viz_configs", [])
            
            if not viz_configs:
                st.info("Gemini didn't recommend specific visuals for this dataset. Here is the default analysis.")
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
                params = config.get("params", {})
                selected_cols = params.get("columns", [])
                reasoning = params.get("reasoning", "")
                
                try:
                    tool_data = json.loads(raw[tool_key]) if tool_key in raw and isinstance(raw[tool_key], str) else raw.get(tool_key)
                    if not tool_data or (isinstance(tool_data, str) and tool_data.startswith("No")):
                        continue

                    # Filter tool_data if columns are specified
                    if selected_cols and isinstance(tool_data, dict):
                        tool_data = {k: v for k, v in tool_data.items() if k in selected_cols}
                        if not tool_data: 
                            st.warning(f"AI requested columns {selected_cols} but they weren't found in tool results.")
                            continue

                    st.write(f"### {title}")
                    if description:
                        st.caption(description)
                    if reasoning:
                        st.info(f"**AI Reasoning:** {reasoning}")

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
                        if selected_cols:
                            corr_df = corr_df[corr_df['feature_a'].isin(selected_cols) | corr_df['feature_b'].isin(selected_cols)]
                        
                        if not corr_df.empty:
                            fig = px.bar(corr_df, x='correlation', y='feature_a', color='feature_b',
                                         orientation='h', title=title)
                            st.plotly_chart(fig, use_container_width=True)

                    elif tool_key == "categorical_analysis":
                        cols = st.columns(min(len(tool_data), 3))
                        for i, (col_name, info) in enumerate(tool_data.items()):
                            with cols[i % 3]:
                                top_v = pd.DataFrame.from_dict(info['top_5_values'], orient='index').reset_index()
                                top_v.columns = ['Value', 'Count']
                                fig = px.pie(top_v, names='Value', values='Count', title=f"{col_name}")
                                st.plotly_chart(fig, use_container_width=True)

                    elif tool_key == "descriptive_stats":
                        stats_df = pd.DataFrame(tool_data).T.reset_index()
                        stats_df.columns = ['Feature', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                        if selected_cols:
                            stats_df = stats_df[stats_df['Feature'].isin(selected_cols)]
                        
                        if not stats_df.empty:
                            fig = px.bar(stats_df, x='Feature', y='mean', error_y='std', title=title)
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    pass

    elif not uploaded_file:
        st.info("Upload a CSV file above to get started.")
