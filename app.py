import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

import os
openai_key = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=None)

st.title("🚙 Emobi - よくある質問チャット")

query = st.text_input("知りたいことを聞いてください", placeholder="例：保険はついてますか？")

if query:
    with st.spinner("調べています..."):
        result = qa_chain.invoke({"query": query})
        answer = result.get("result", "回答が見つかりませんでした。")
    st.markdown(f"### ✅ 回答\n{answer}")
