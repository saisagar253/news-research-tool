import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()

# ---------------- UI ----------------
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()

# ---------------- LLM (Groq) ----------------
llm = ChatGroq(
    temperature=0.3,
    model_name="llama3-70b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- Process URLs ----------------
if process_url_clicked and urls:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data... ✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    main_placeholder.text("Creating embeddings... ✅")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.text("Processing completed 🚀")
    time.sleep(1)

# ---------------- Question Answering ----------------
query = st.text_input("Question:")

if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    result = chain({"question": query}, return_only_outputs=True)

    st.header("Answer")
    st.write(result["answer"])

    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources")
        for src in sources.split("\n"):
            st.write(src)
