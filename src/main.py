from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from .document_processor import process_files
from .rag_engine import query_rag_system
from langchain_community.embeddings import SentenceTransformerEmbeddings

def initialize_rag_system(api_key):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=api_key,
        temperature=0.7
    )

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    combined_vectorstore, all_sql_databases = process_files(embedding_model)
    return combined_vectorstore, all_sql_databases, llm

def get_rag_response(combined_vectorstore, all_sql_databases, query, llm):
    return query_rag_system(combined_vectorstore, all_sql_databases, query, llm)