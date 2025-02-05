from typing import TypedDict, List, Union, Dict, Any
from langgraph.graph import StateGraph, END
from langchain.schema import AIMessage, HumanMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.agent_toolkits import create_sql_agent

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    input: str  # Changed from current_input to input
    context: str
    answer: str

def create_vectorstore_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever()
    
    prompt = ChatPromptTemplate.from_template("""
        As Razi, AI&You's analytical assistant specializing in technical documentation analysis. Your main purpose
        is to answer the user query. Here is a series of guidelines you need to follow:
        
        Process:
        1. Context Analysis: Review {messages} for relevant technical context
        2. Question Evaluation: Assess "{input}" against {context}
        3. Structured Reasoning:
           - Identify key concepts and relationships
           - Reference specific documentation examples
           - Highlight assumptions made
        4. Resolution: Provide either:
           - Complete answer with supporting evidence, or
           - Clear identification of missing data with impact analysis
        
        Examples of good responses:
        "Based on API documentation v2.3, the correct endpoint is..."
        "The context doesn't contain error code definitions - needed to diagnose..."
        
        Ask clarifying questions if: 
        - Question scope is ambiguous
        - Multiple interpretations exist
        - Security implications might be involved
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def create_sql_chain(db, llm):
    base_prompt = PromptTemplate(
        input_variables=["messages"],
        template="""
        While answering the user query, follow this process:
        1. Review conversation history ({message}) for relevant context
        2. Analyze the SQL schema and available tables
        3. Explain your reasoning step-by-step
        4. Generate final SQL query

        SQL Query Generation Protocol:
        
        1. Context Review: Analyze {messages} for:
           - Filter patterns
           - Date ranges
           - Aggregation requirements
         
        2. Schema Analysis: Verify table relationships using:
           - Primary/Foreign key mappings
           - Indexed columns
           - Data type compatibility
         
        3. Query Construction:
           - Specify output format (raw/aggregated)
           - Include EXPLAIN ANALYZE for validation
           - Add error handling (TRY/CATCH patterns)
         
        4. Validation Checklist:
           - Cross-reference schema version
           - Test JOIN conditions
           - Verify NULL handling
         
        Return format:
        - Markdown-formatted explanation
        - Valid SQL query
        - Potential alternatives if performance concerns exist
        """
    )

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type='openai-tools',
        verbose=True,
        prompt=base_prompt
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