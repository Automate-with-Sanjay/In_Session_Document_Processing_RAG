import os
from dotenv import load_dotenv

# Load environment variables FIRST before anything else happens
load_dotenv()
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader # Or CSV/Pandas if you prefer row-by-row
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import OpenAIEmbeddings



# 1. Load your data
# # This could easily be swapped for a PDFLoader or a directory loader
# loader = TextLoader("your_data.txt")
# docs = loader.load()

# text = """I am Spidy,Your very long document text goes here. 
# It might have multiple paragraphs. You want to chunk it nicely."""


FILE_PATH = r"C:\Users\skbin\Downloads\Coding Assessment answer.txt"


text = loader.load()
print("Loaded text:", text[:500])  # Print the first 500 characters to verify loading

# 2. Chunk the data
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# splits = text_splitter.split_documents(docs)

# Initialize the splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # Max number of characters per chunk
    chunk_overlap=20,    # Overlap between chunks to maintain context
    length_function=len,
    is_separator_regex=False,
)
# Split the text into a list of strings
chunks = text_splitter.split_text(text)
print("Text chunks:", chunks)

# Alternatively, split into LangChain Document objects
docs = text_splitter.create_documents([text])
print ("chunks:", docs)

# 3 & 4. Embed and store in a local vector database
# Using nomic-embed-text via Ollama for fast, local embeddings
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=OllamaEmbeddings(model="nomic-embed-text")
    
)
retriever = vectorstore.as_retriever()

# 5. Set up the local LLM (e.g., llama3.2:3B via Ollama)
llm = ChatOllama(model="qwen3.5:0.8b")

# 6. Define the Prompt Template
template = """Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. Construct the RAG Chain
rag_chain = (
    {"context": retriever | format_docs, 
     "question": RunnablePassthrough()
     }
    | prompt
    | llm
    | StrOutputParser()
)

# Run the query
# ... (Keep your prompt, format_docs, and rag_chain definition exactly the same as before) ...

print("🤖 Assistant: Hello! Ask me anything about your documents. (Type 'exit' to quit)\n")

# Start an infinite loop for real-time interaction
while True:
    # 1. Capture the user's input in real-time
    user_question = input("🙋 You: ")
    
    # 2. Check if the user wants to break out of the chat
    if user_question.lower() in ['exit', 'quit', 'q']:
        print("🤖 Assistant: Goodbye!")
        break
        
    # 3. Skip empty inputs if the user accidentally just hits enter
    if not user_question.strip():
        continue
        
    try:
        # 4. Run the live question through your RAG pipeline
        print("🤖 Assistant is thinking...")
        response = rag_chain.invoke(user_question)
        
        # 5. Print the real-time response
        print(f"\n🤖 Assistant: {response}\n" + "-"*40 + "\n")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}\n")