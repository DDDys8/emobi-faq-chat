import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# ✅ APIキーをsecretsから安全に読み込む
openai_key = st.secrets["OPENAI_API_KEY"]

# ✅ LLM（GPT-3.5）を初期化
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_key,
    temperature=0.3
)

# ✅ ベクトルDBの読み込み（chroma_store は GitHub 上に配置済）
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embedding_model
)

# ✅ Retrieverの設定
retriever = db.as_retriever(search_kwargs={"k": 3})

# ✅ QAチェーンを構築（GPT + 検索）
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ✅ Streamlit UI
st.set_page_config(page_title="Emobi AIチャット - FINUA", page_icon="🚙")

st.title("🚙 Emobi AIチャット - FINUA")
st.write("Emobiに関するよくある質問にお答えします。")

query = st.text_input("ご質問をどうぞ")

if query:
    result = qa_chain.invoke({"query": query})

    st.subheader("📌 回答")
    st.write(result["result"])

    st.subheader("📚 参照されたFAQ")
    for i, doc in enumerate(result["source_documents"]):
        st.markdown(f"**チャンク{i+1}**\n\n```\n{doc.page_content[:500]}\n```")

from langchain_community.embeddings import HuggingFaceEmbeddings

# CPU 強制指定（Streamlit Cloud ではGPU非対応のため）
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

