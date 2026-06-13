import streamlit as st
import ollama

# ============================================
# Configuration
# ============================================

OLLAMA_HOST = "http://localhost:11434"

MODELS = [
    "gemma3:4b",
    "gemma3:12b",
    "qwen2.5vl:7b",
    "llava:7b"
]

SYSTEM_PROMPT = """
You are an advanced multimodal AI assistant.

You carefully analyze uploaded images and answer questions about them.

Rules:

- Examine the image carefully.
- Identify objects, people, animals, colors, surroundings and text.
- Perform OCR whenever visible text exists.
- Use only information visible in the image.
- Never hallucinate.
- If uncertain, explicitly mention uncertainty.
- Explain scenes naturally and conversationally.
- Remember previous conversation.
- Behave like ChatGPT for images.
- Provide detailed and helpful responses.
"""

client = ollama.Client(host=OLLAMA_HOST)

# ============================================
# Page config
# ============================================

st.set_page_config(
    page_title="Gemma3 Image Chatbot",
    page_icon="🖼️",
    layout="wide"
)

# ============================================
# Session State
# ============================================

def initialize_session():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None


# ============================================
# Sidebar
# ============================================

def sidebar():

    with st.sidebar:

        st.title("Settings")

        selected_model = st.selectbox(
            "Choose Model",
            MODELS
        )

        temperature = st.slider(
            "Temperature",
            0.0,
            2.0,
            0.7,
            0.1
        )

        if st.button("🗑 Clear Chat"):
            st.session_state.messages = []

        return selected_model, temperature


# ============================================
# Upload image
# ============================================

def image_uploader():

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:

        st.session_state.uploaded_image = uploaded_file

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )


# ============================================
# Display previous chat
# ============================================

def display_chat():

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


# ============================================
# Build messages
# ============================================

def build_messages(query):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(st.session_state.messages)

    current_message = {
        "role": "user",
        "content": query
    }

    # Attach image to current question
    if st.session_state.uploaded_image is not None:

        image_bytes = st.session_state.uploaded_image.getvalue()

        current_message["images"] = [image_bytes]

    messages.append(current_message)

    return messages


# ============================================
# Stream response
# ============================================

def stream_response(messages, model, temperature):

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        stream = client.chat(
            model=model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature
            }
        )

        for chunk in stream:

            token = chunk["message"]["content"]

            full_response += token

            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    return full_response


# ============================================
# Main
# ============================================

def main():

    st.title("🖼️ Image Chatbot with Ollama")

    st.markdown(
        """
        Upload an image and ask questions about it.

        - Image memory
        - Streaming responses
        - Multiple models
        - ChatGPT style interface
        """
    )

    initialize_session()

    selected_model, temperature = sidebar()

    image_uploader()

    display_chat()

    query = st.chat_input(
        "Ask something about the image..."
    )

    if query:

        # Show user message
        with st.chat_message("user"):
            st.markdown(query)

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        messages = build_messages(query)

        answer = stream_response(
            messages,
            selected_model,
            temperature
        )

        # Save assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================
# Run app
# ============================================

if __name__ == "__main__":
    main()
