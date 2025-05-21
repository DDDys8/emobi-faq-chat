import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ✅ OpenAI APIキーを .streamlit/secrets.toml に記載してください
openai_key = st.secrets["OPENAI_API_KEY"]

# ✅ 埋め込みモデル（軽量モデル＆CPU動作）
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ LLM（OpenAI Chat API）
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)

# ✅ よくある質問のペア（ここを編集）
faq_pairs = [
    "Q: 初期費用は合計でいくらかかるのか？\nA: 実質輸送費となります。",
    "Q: 保険の内容はどうなっていますか？\nA: 任意保険に加入済で、対人・対物・搭乗者全てカバーされます。",
    "Q: 1日の平均稼働回数は？\nA: 平日は1回転、休日は2回転が目安です。",
    "Q: マーケティング支援って本当に効果あるの？\nA: SNSを軸に実績があり、集客増加に貢献しています。",
    "Q: 故障や事故時の対応はどうなっていますか？\nA: サポートチームが営業時間内で即時対応します。"
]

# ✅ FAQデータをベクトル化してFAISSに格納
db = FAISS.from_texts(texts=faq_pairs, embedding=embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 4})

# ✅ LangChainのRetrievalQAチェーン
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ✅ Streamlit UI構築
st.set_page_config(page_title="Emobiチャット", page_icon="🤖")
st.title("✅ Emobi チャット - よくあるご質問")

query = st.text_input("気になることを質問してみてください（例：保険はどうなってますか？）")

if query:
    with st.spinner("考え中です..."):
        result = qa_chain.invoke({"query": query})
        st.markdown("### 📌 回答：\n" + result["result"])
