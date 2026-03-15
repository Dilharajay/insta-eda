"""
InstaEDA — Streamlit UI
Run with: streamlit run app.py
"""

import streamlit as st
from utils.auth import init_db

from ui.sidebar import render_sidebar
from ui.run_tab import render_run_tab
from ui.settings_tab import render_settings_tab
from ui.account_tab import render_account_tab

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
# Sidebar / Authentication
# ─────────────────────────────────────────────
render_sidebar()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("⚡ InstaEDA")
st.caption("Drop in a CSV. Get a full EDA report + AI-recommended visuals — powered by Gemini.")
st.divider()

if not st.session_state.get("authenticated", False):
    st.info("Please login or register in the sidebar to start using InstaEDA.")
    st.stop()

# ─────────────────────────────────────────────
# Main Application Flow
# ─────────────────────────────────────────────

# Tabs — Run, Settings, and Account
tab_run, tab_settings, tab_account = st.tabs(["🚀 Analysis", "⚙️ API Settings", "👤 Account Management"])

with tab_account:
    render_account_tab()

with tab_settings:
    render_settings_tab()

with tab_run:
    render_run_tab()
