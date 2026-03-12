import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate



# ------------------ AYARLAR ------------------
st.set_page_config(page_title="Eczacı Asistanı", page_icon="💊")
st.title("💊 Akıllı İlaç & Prospektüs Asistanı")

load_dotenv()

DATA_DIR = "./data"
DB_DIR = "./chroma_db_ilac"


# ------------------ RAG SETUP ------------------
@st.cache_resource
def setup_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        st.error("data klasörü yoktu, oluşturuldu. PDF ekleyip yeniden başlat.")
        st.stop()

    if not os.path.exists(DB_DIR):
        with st.spinner("PDF'ler işleniyor..."):
            loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
            docs = loader.load()

            if not docs:
                st.error("data klasöründe PDF yok.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=300
            )
            splits = splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                splits, embedding, persist_directory=DB_DIR
            )
    else:
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding
        )

    return vectorstore.as_retriever(search_kwargs={"k": 15})

# ------------------ LLM ------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)



prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sen bir eczacı asistanısın. "
        "Sadece verilen dökümanlara dayanarak çıkarım yap. "
        "Dökümanda yazmıyor olabilir ama verilen farmakolojik mekanizmalardan bilimsel çıkarım yap, ama çıkarım yaptığında bunun çıkarım olduğunu söyle."
        "Eğer belgede ilgili mekanizma tamamen yoksa, açıkça ‘bu konuda çıkarım yapılamaz’ de."
    ),
    (
        "human",
        "Soru: {question}\n\n"
        "Bağlam:\n{context}"
    )
])

# ------------------ SORU CEVAP ------------------
def answer_question(question: str, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm
    response = chain.invoke({
        "question": question,
        "context": context
    })

    return response.content

# ------------------ UI ------------------
retriever = setup_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("İlaç sorunuzu yazın (örn: Parol günde kaç kere içilir?)"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Araştırılıyor..."):
            try:
                answer = answer_question(user_input, retriever)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"Hata: {e}")
