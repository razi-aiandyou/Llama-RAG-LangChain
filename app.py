import streamlit as st
import os
from dotenv import load_dotenv
from src.main import initialize_rag_system, get_rag_response
from src.file_manager import save_uploaded_file, get_uploaded_files, remove_file, clear_upload_folder

# Load environment variables
load_dotenv()

# set page configuration
st.set_page_config(
    page_title="Llama RAG Developed by Razi Ashary",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Dark theme adjustments */
    body {
        color: #E0E0E0;
        background-color: #0E1117;
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Roboto', sans-serif;
        font-size: 3em;
        font-weight: 700;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1em;
        text-shadow: 2px 2px 4px #000000;
    }
    
    /* Subheader styling */
    .sub-header {
        font-size: 1.5em;
        color: #4169E1;
        text-align: center;
        margin-bottom: 2em;
    }
    
    /* Chat container styling */
    .chat-container {
        border: 2px solid #4169E1;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #1E1E1E;
    }
    
    /* User message styling */
    .user-message {
        background-color: #2C3E50;
        color: #FFFFFF;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-end;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #34495E;
        color: #FFFFFF;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-start;
    }
    
    /* Input box styling */
    .stTextInput>div>div>input {
        background-color: #2C3E50;
        color: #FFFFFF;
        border: 1px solid #4169E1;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4169E1;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1E1E1E;
    }
</style>
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

# Main header
st.markdown("<h1 class='main-header'>🦙 Llama RAG by razi</h1>", unsafe_allow_html=True)

# Subheader
st.markdown("<p class='sub-header'>Your AI-powered knowledge assistant</p>", unsafe_allow_html=True)

# Chat container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='{message['role']}-message'>{message['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# React to user input
prompt = st.text_input("Ask me anything:", key="user_input")
if st.button("Send", key="send_button"):
    if prompt:
        # Display user message in chat message container
        st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        combined_vectorstore, all_sql_databases, llm = st.session_state.rag_system
        response = get_rag_response(combined_vectorstore, all_sql_databases, prompt, llm)

        # Display assistant response in chat message container
        st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("""
<div style='position: fixed; bottom: 0; left: 0; right: 0; text-align: center; padding: 10px; background-color: #1E1E1E;'>
    <p style='color: #4169E1;'>Powered by LangChain and Groq | Developed by razi</p>
</div>
""", unsafe_allow_html=True)
