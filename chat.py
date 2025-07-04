import streamlit as st
import os
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="📊  AI Research Assistant for Papers & PDFs", layout="wide")
st.title("📊  AI Research Assistant for Papers & PDFs")
st.markdown("Let users upload papers and ask things like “What’s the main contribution?” or “Summarize section 3.1.")

# Upload PDF
uploaded_file = st.file_uploader("📎 Upload Budget Speech PDF", type="pdf")
query = st.text_input("🔍 Ask a question about the budget")

# Load API keys from environment
gemini_key = os.getenv("GOOGLE_API_KEY", "")
pinecone_key = os.getenv("PINECONE_API_KEY", "")
index_name = "rag-app"

if uploaded_file and query:
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split documents
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Set Pinecone API key
    os.environ['PINECONE_API_KEY'] = pinecone_key

    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key=gemini_key,
        task_type="retrieval_query"
    )

    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_key,
        temperature=0.3,
        safety_settings=safety_settings
    )

    # Build vector DB
    with st.spinner("🔗 Creating vector store..."):
        vectordb = PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=index_name
        )

    # Prompt template
    prompt_template = """
    You are an intelligent assistant specialized in analyzing and summarizing Indian Budget speeches and financial documents.

    ## Instructions:
    Use the **context** provided below to answer the **question** clearly and factually. Focus only on the content from the document — do not generate your own opinions or assumptions.

    If the answer is **not found in the context**, respond with:
    **"The answer is not available in the provided document."**

    ## Additional Guidelines:
    - Avoid political bias or speculation
    - Focus on budgetary figures, schemes, reforms, and policy announcements
    - Do not infer or hallucinate numbers or statements

    ---

    ### Context:
    {context}

    ### Question:
    {question}

    ### Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        llm=chat_model
    )

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    # Run query
    with st.spinner("🧠 Generating answer..."):
        response = qa_chain.invoke({"query": query})

    st.subheader("📘 Answer")
    st.write(response["result"])

    st.markdown("---")
    st.subheader("📄 Source Chunks")
    for i, doc in enumerate(response['source_documents']):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(doc.page_content)
        st.markdown("---")

elif not uploaded_file:
    st.info("📎 Please upload a Budget Speech PDF to get started.")
elif not query:
    st.info("💬 Please enter a question to analyze the uploaded budget.")