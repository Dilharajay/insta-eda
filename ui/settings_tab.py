import streamlit as st
from utils.auth import get_api_key, save_api_key

def render_settings_tab():
    st.header("⚙️ Configuration")
    
    # Retrieve existing API key
    existing_key = get_api_key(st.session_state["username"])
    
    api_key_input = st.text_input(
        "Google API Key",
        value=existing_key if existing_key else "",
        type="password",
        placeholder="AIza...",
        help="Your key is stored securely in the database for your next session.",
        key="api_key_input"
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
        key="model_name"
    )
    
    st.divider()
    st.markdown(f"**Account:** `{st.session_state['username']}`")
    st.markdown(f"**Current Model:** `{model_name}`")
    st.markdown("**Tools:** 7 automated EDA tools")
    st.markdown("**AI Visuals:** Smart Chart Selection")
