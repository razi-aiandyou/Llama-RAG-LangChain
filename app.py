import streamlit as st
import os
from dotenv import load_dotenv
from src.main import initialize_rag_system, get_rag_response
from src.file_manager import save_uploaded_file, get_uploaded_files, remove_file, clear_upload_folder
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# set page configuration
st.set_page_config(
    page_title="Llama RAG Developed by Razi Ashary",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Global Styling */
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f5f5f5;
    }

    /* Header */
    .main-header {
        text-align: center;
        font-size: 2em;
        color: #FFD700;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .sub-header {
        text-align: center;
        font-size: 1.2em;
        color: #7f8c8d;
        margin-bottom: 20px;
    }

    /* Chat Window */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(70vh - 20px);
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        margin: 0 auto;
    }

    .message-bubble {
        margin: 10px;
        padding: 15px;
        border-radius: 10px;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .user-message {
        align-self: flex-end;
        background-color: #3498db;
        color: white;
        border-radius: 10px 10px 0 10px;
    }

    .assistant-message {
        align-self: flex-start;
        background-color: #2c3e50;
        color: white;
        border-radius: 10px 10px 10px 0;
    }

    /* Input Section */
    .input-container {
        display: flex;
        gap: 10px;
        align-items: center;
        padding: 10px;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f5f5f5;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
    }

    .input-container input[type='text'] {
        flex: 1;
        padding: 10px;
        border-radius: 25px;
        border: 1px solid #ccc;
        font-size: 1em;
    }

    .input-container button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 8px 20px;
        font-size: 1em;
        cursor: pointer;
    }

    .input-container button:hover {
        background-color: #2874a6;
    }

    /* Footer */
    .footer-container {
        text-align: center;
        padding: 10px;
        background-color: #1E1E1E;
        color: #4169E1;
        font-size: 0.9em;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
    }

    /* Responsive Adjustments */
    @media (max-width: 768px) {
        .chat-container {
            margin: 0 10px;
        }

        .message-bubble {
            max-width: 90%;
        }
    }
</style>
""", unsafe_allow_html=True)

# JavaScript for Enter key submission
st.markdown("""
<script>
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            const button = document.querySelector('button[kind="primary"]');
            if (button) {
                button.click();
            }
        }
    });
</script>
""", unsafe_allow_html=True)

# Sidebar for file and upload management
st.sidebar.title("File Management")

uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "csv", "txt", "md", "json"])
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.sidebar.success(f"File {uploaded_file.name} uploaded sucessfully")

    # Reinitialize RAG system
    st.session_state.rag_system = initialize_rag_system(os.getenv("GROQ_API_KEY"))
    st.sidebar.success("File processed for Question Answering!")

# Display uploaded files
st.sidebar.subheader("Uploaded Files")
uploaded_files = get_uploaded_files()
for file in uploaded_files:
    col1, col2 = st.sidebar.columns([3, 1])
    col1.write(file)
    if col2.button("Remove", key=f"remove_{file}"):
        remove_file(file)
        st.sidebar.success(f"File {file} removed successfully!")
        st.session_state.rag_system = initialize_rag_system(os.getenv("GROQ_API_KEY"))
        st.rerun()

if st.sidebar.button("Clear All Files"):
    clear_upload_folder()
    st.sidebar.success("All files cleared successfully!")
    st.session_state.rag_system = initialize_rag_system(os.getenv("GROQ_API_KEY"))
    st.rerun()

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = initialize_rag_system(os.getenv("GROQ_API_KEY"))
    st.session_state.messages = []

# Main header
st.markdown("<h1 class='main-header'>🦙 Llama RAG by razi</h1>", unsafe_allow_html=True)

# Subheader
st.markdown("<p class='sub-header'>Your AI-powered knowledge assistant</p>", unsafe_allow_html=True)

# Chat container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    message_type = "user" if isinstance(message, HumanMessage) else "assistant"
    st.markdown(
        f"<div class='message-bubble {message_type}-message'>{message.content}</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# Fixed input container at the bottom
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

# Input container
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
prompt = st.text_input("", placeholder="Type your message...", key="user_input")
send_button = st.button("Send", key="send_button")
st.markdown("</div>", unsafe_allow_html=True)

if send_button and prompt:
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    
    response = get_rag_response(
        st.session_state.rag_system,
        prompt,
        st.session_state.messages
    )
    
    assistant_message = AIMessage(content=response)
    st.session_state.messages.append(assistant_message)
    st.rerun()

# Footer
st.markdown("""
<div class='footer-container'>
    <p>Powered by LangChain and Groq | Developed by Razi</p>
</div>
""", unsafe_allow_html=True)