import streamlit as st
import ollama

# Initialize client
client = ollama.Client("http://localhost:11434")
MODEL_NAME = "gemma3:4b"

# Set page config
st.set_page_config(page_title="Image Chatbot", page_icon="🤖", layout="centered")

# Title
st.title("🖼️ Image Chatbot with Gemma3")
st.markdown("Upload an image, ask questions about it, and get answers!")

# Instructions box with shadow effect
st.markdown(
    """
    <div style="padding:10px; box-shadow: 3px 3px 10px #ccc; border-radius:10px; background-color:#f9f9f9">
    1. Upload an image.<br>
    2. Ask questions about the image.<br>
    3. Keep asking more questions without losing previous chats.
    </div>
    """, unsafe_allow_html=True
)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg","pdf"])

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# If image uploaded
if uploaded_file:
    # Display the image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Text input for question
    question = st.text_input("Your Question:")

    if st.button("Ask") and question:
        # Build prompt for Gemma3
        prompt = f"You are a helpful assistant that can analyze the image and answer the query: {question}"

        # Save question in history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Call Ollama API
        response = client.generate(MODEL_NAME, prompt, images=[uploaded_file], stream=False)
        answer = response.get("response", "No response from model.")

        # Save answer in history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['content']}")
        else:
            st.markdown(f"**Bot:** {chat['content']}")

