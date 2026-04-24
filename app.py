import os
import streamlit as st
import pdfplumber

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)

st.set_page_config(page_title="GenAI Chatbot", layout="wide")
st.header("GenAI Chatbot")

api_key = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if st.button("Process"):
        if not api_key:
            st.error("OpenAI API key not found.")
        elif uploaded_file is not None:
            text = ""

            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_text(text)

            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

            st.session_state["vectorstore"] = vectorstore
            st.session_state["chunks"] = chunks

            st.success(f"PDF processed successfully. Total chunks: {len(chunks)}")
        else:
            st.warning("Please upload a PDF first.")

user_question = st.text_input("Ask a question about the PDF")

if user_question:
    if "vectorstore" not in st.session_state:
        st.warning("Please upload and process a PDF first.")
    else:
        docs = st.session_state["vectorstore"].similarity_search(user_question, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{user_question}

Answer clearly:
"""

        llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-5.4-mini"
        )

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

if "chunks" in st.session_state:
    st.subheader("Chunk Preview")
    st.write(st.session_state["chunks"][0])
else:
    st.write("Chatbot interface is ready.")