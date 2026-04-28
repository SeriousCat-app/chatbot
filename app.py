from snowflake_utils import run_snowflake_query
import os
import streamlit as st
import pdfplumber

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


st.set_page_config(page_title="GenAI Chatbot", layout="wide")

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

st.header("GenAI Chatbot")

mode = st.sidebar.selectbox(
    "Choose Mode",
    ["PDF Chatbot", "Snowflake Insights"]
)


if mode == "PDF Chatbot":

    with st.sidebar:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

        if st.button("Process", key="pdf_process_btn"):
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
            docs = st.session_state["vectorstore"].similarity_search(
                user_question,
                k=3
            )

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


elif mode == "Snowflake Insights":

    st.subheader("Snowflake Business Insights")

    st.write(
        "This mode runs a real SQL query against Snowflake and then uses AI "
        "to generate business insights from the result."
    )

    query = """
    select
        year(review_date) as review_year,
        review_sentiment as neg_neut_pos,
        is_full_moon as full_not_full_moon,
        count(*) as reviews_cnt
    from mart_fullmoon_reviews
    where review_date between '2020-01-01' and '2021-12-31'
    group by 1,2,3
    order by 1,2,3
    """

    if st.button("Run Snowflake Query", key="run_snowflake_query_btn"):

        try:
            df = run_snowflake_query(query)

            st.session_state["snowflake_df"] = df

            st.success("Snowflake query completed successfully.")
            st.subheader("Snowflake Data Preview")
            st.dataframe(df)

        except Exception as e:
            st.error("Snowflake query failed.")
            st.exception(e)

    if "snowflake_df" in st.session_state:

        df = st.session_state["snowflake_df"]

        st.subheader("Current Snowflake Result")
        st.dataframe(df)

        if st.button("Generate AI Insights", key="generate_snowflake_insights_btn"):

            if not api_key:
                st.error("OpenAI API key not found.")
            else:
                prompt = f"""
You are a senior business analyst.

Analyze the following Snowflake result and provide:
1. Key trends
2. Differences between full moon and not full moon periods
3. Differences between positive, neutral and negative review sentiment
4. Risks or anomalies
5. Recommended business actions

Data:
{df.to_string(index=False)}

Write the answer in a clear business style.
"""

                llm = ChatOpenAI(
                    openai_api_key=api_key,
                    model="gpt-5.4-mini"
                )

                response = llm.invoke(prompt)

                st.subheader("AI Business Insights")
                st.write(response.content)