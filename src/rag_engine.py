from typing import TypedDict, List, Union, Dict, Any
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import create_sql_agent

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    input: str  # Changed from current_input to input
    context: str
    answer: str

def create_vectorstore_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever()
    
    prompt = ChatPromptTemplate.from_template("""
        You are Razi, an analyst created by AI&You. Your task is to answer the following 
        question based on the provided context and conversation history. If the provided context
        does not provide enough information to answer the question, gently state you
        don't know and explain the reason.

        Current conversation history:
        {messages}

        Question: {input}
        Context: {context}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def create_sql_chain(db, llm):
    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type='openai-tools',
        verbose=True
    )

def process_with_vectorstore(
    state: Dict[str, Any],
    vectorstore,
    llm
) -> Dict[str, Any]:
    chain = create_vectorstore_chain(vectorstore, llm)
    # Use state directly as it now has the correct 'input' key
    response = chain.invoke(state)
    state["answer"] = response["answer"]
    return state

def process_with_sql(
    state: Dict[str, Any],
    all_sql_databases,
    llm
) -> Dict[str, Any]:
    for db in all_sql_databases.values():
        try:
            chain = create_sql_chain(db, llm)
            response = chain.invoke(state)  # Pass state directly
            if response:
                state["answer"] = response["output"]
                return state
        except Exception as e:
            print(f"Error querying database: {str(e)}")
            continue
    return state

def should_use_sql(state: Dict[str, Any]) -> bool:
    return any(phrase in state["answer"] for phrase in [
        "I don't have enough information",
        "I'm not sure",
        "I don't know"
    ])

def create_rag_graph(vectorstore, all_sql_databases, llm):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("vectorstore", lambda x: process_with_vectorstore(x, vectorstore, llm))
    workflow.add_node("sql", lambda x: process_with_sql(x, all_sql_databases, llm))
    
    workflow.set_entry_point("vectorstore")
    
    workflow.add_conditional_edges(
        "vectorstore",
        should_use_sql,
        {
            True: "sql",
            False: END
        }
    )
    
    workflow.add_edge("sql", END)
    
    return workflow.compile()