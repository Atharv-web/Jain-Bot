import streamlit as st
from jb import chatbot

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="🕉 JainBot",
    page_icon="🕉",
    layout="centered"
)

st.title("Jain-Bot")

# --- Session State for Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat Input ---
user_input = st.chat_input("Ask your questions...")

if user_input:
    # Store user query
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        try:
            # Run chatbot (synchronous function from jain_agent.py)
            answer = chatbot(user_input)
        except Exception as e:
            answer = f"⚠️ Server error: {e}"

    # Store bot response
    st.session_state.chat_history.append(("bot", answer))

# --- Render Chat Messages ---
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)