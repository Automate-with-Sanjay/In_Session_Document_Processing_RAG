# src/query.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Check your .env file.")

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()

DB_DIR = str(DATA_DIR / "chroma_db")
# os.makedirs(DB_DIR, exist_ok=True)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def answer_user_question(question: str, session_id: str):
    """Retrieves context specific to the session_id and generates an answer."""
    # 1. Access the existing vector database
    vector_store = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2", 
            google_api_key=API_KEY
        )
    )
    
    # 2. Setup retriever to ONLY pull documents matching this user session
    retriever = vector_store.as_retriever(
        search_kwargs={"filter": {"session_id": session_id}, "k": 3}
    )
    
    # 6. Define the Prompt Template
    template = """Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Context: {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)


    # 4. Define LLM for response generation
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0, google_api_key=API_KEY)
    
    rag_chain = (
        {"context": retriever | format_docs,
        "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            )
    
    # Get response
    
    return rag_chain.invoke(question)

