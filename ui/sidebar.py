import streamlit as st
import json
from utils.auth import (
    register_user, authenticate_user, 
    get_user_reports, delete_report
)

def render_sidebar():
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
