import streamlit as st
import ollama

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Offline AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Offline AI Assistant")
st.caption("Powered by Ollama + Gemma3:4b")

# ---------------------------
# System Prompt
# ---------------------------
SYSTEM_PROMPT = """
You are an offline AI assistant running locally.

Guidelines:
- Provide concise and useful answers.
- Use bullet points whenever appropriate.
- Structure responses clearly.
- Give step-by-step explanations if needed.
- Be helpful and professional.
- If code is requested, provide clean and commented code.
"""

# ---------------------------
# Initialize Chat Memory
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

# ---------------------------
# Display Previous Messages
# ---------------------------
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------------------------
# User Input
# ---------------------------
user_query = st.chat_input("Ask me anything...")

if user_query:

    # Store user query
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # Display user query
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        try:
            response = ollama.chat(
                model="gemma3:4b",
                messages=st.session_state.messages
            )

            assistant_reply = response["message"]["content"]

        except Exception as e:
            assistant_reply = f"❌ Error: {str(e)}\n\nMake sure Ollama is running."

        response_placeholder.markdown(assistant_reply)

    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("Settings")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        st.rerun()

    st.info(
        f"Conversation turns: "
        f"{len(st.session_state.messages)-1}"
    )
