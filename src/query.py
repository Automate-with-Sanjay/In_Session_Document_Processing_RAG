# src/query.py
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = "./chroma_db"

def answer_user_question(question: str, session_id: str):
    """Retrieves context specific to the session_id and generates an answer."""
    # 1. Access the existing vector database
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=OpenAIEmbeddings())
    
    # 2. Setup retriever to ONLY pull documents matching this user session
    retriever = vector_store.as_retriever(
        search_kwargs={"filter": {"session_id": session_id}, "k": 3}
    )
    
    # 3. Build the prompt template
    system_prompt = (
        "You are an expert assistant. Answer the user's question using ONLY the provided context. "
        "If you do not know the answer, say you don't know.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 4. Link LLM and Vector Retriever together
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 5. Get response
    response = rag_chain.invoke({"input": question})
    return response["answer"]