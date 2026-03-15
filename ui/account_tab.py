import streamlit as st
from utils.auth import update_username, update_password, authenticate_user

def render_account_tab():
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
