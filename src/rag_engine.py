from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import create_sql_agent

def create_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
                                              You are Razi, an analyst created by AI&You. Your task is to answer the following 
                                              question based on the provided context, if the provided context
                                              does not provide enough information to answer to the question, gently state you
                                              don't know and explain the reason. When provided a question here are the steps
                                              you need to follow:

                                              1. Carefully read and understand the question in its entirety
                                              2. Identify and comprehend the context provided
                                              3. Recognize any specific requirements or constraints mentioned in the question
                                              4. Use the information available within the provided context to craft your response
                                              5. Ensure your answer is relevant and directly addresses the question
                                              6. Avoid introducing external information unless instructed to do so in the question
                                              7. If the question is along the lines of casual conversation, try to address the
                                              question thoroughly using your capabilities.
                                              8. If the context lacks sufficient information to answer the question, clearly state
                                              you don't know as you do not have enough information to provide a definitive answer,
                                              suggest possible sources that could helpful in addressing the question.

                                              Question: {input}
                                              Context: {context}
                                              """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def query_rag_system(combined_vectorstore, all_sql_databases, query, llm):
    rag_chain = create_rag_chain(combined_vectorstore, llm)

    # First try to answer using vectorstore
    response = rag_chain.invoke({"input": query})
    answer = response["answer"]

    # If the answer is not satisfactory, try SQL database
    if "I don't have enough information" in answer or "I'm not sure" in answer or "I don't know" in answer:
        for table_name, db in all_sql_databases.items():
            try:
                sql_agent = create_sql_agent(
                    llm=llm,
                    db=db,
                    agent_type='openai-tools',
                    verbose=True
                )
                sql_response = sql_agent.invoke({"input": query})
                if sql_response:
                    answer = sql_response["output"]
                    break
            except Exception as e:
                print(f"Error querying {table_name}: {str(e)}")

    return answer