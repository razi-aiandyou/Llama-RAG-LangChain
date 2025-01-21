from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from typing import Union, List
from .document_processor import process_files
from .rag_engine import create_rag_graph
from langchain_community.embeddings import SentenceTransformerEmbeddings

def initialize_rag_system(api_key):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=api_key,
        temperature=0.7
    )

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    combined_vectorstore, all_sql_databases = process_files(embedding_model)
    
    rag_graph = create_rag_graph(combined_vectorstore, all_sql_databases, llm)
    return rag_graph

def get_rag_response(rag_graph, query: str, messages: List[Union[HumanMessage, AIMessage]]) -> str:
    # Initialize state with 'input' instead of 'current_input'
    state = {
        "messages": messages,
        "input": query,  # Changed key to 'input'
        "context": "",
        "answer": ""
    }
    
    result = rag_graph.invoke(state)
    return result["answer"]
