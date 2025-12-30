import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Kütüphaneleri
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_core.prompts import PromptTemplate

# 1. AYARLAR
st.set_page_config(page_title="Eczacı Asistanı", page_icon="💊")
st.title("💊 Akıllı İlaç & Prospektüs Asistanı")
st.markdown("*Ben prospektüsleri okuyan ve dozaj/yan etki hesabı yapabilen bir yapay zeka ajanıyım.*")

load_dotenv()

DB_YOLU = "./chroma_db_ilac"
DATA_YOLU = "./data"


# 2. VERİTABANI VE TOOL OLUŞTURMA (RAG KISMI)
@st.cache_resource
def setup_knowledge_base():
    """
    Bu fonksiyon ilaç PDF'lerini okur ve Ajanın kullanabileceği bir 'Tool' (Alet) haline getirir.
    """
    if not os.path.exists(DB_YOLU):
        with st.spinner("İlaç prospektüsleri taranıyor ve veritabanı kuruluyor..."):
            # KLASÖRDEKİ TÜM PDF'LERİ YÜKLE (DirectoryLoader farkı burada!)
            loader = DirectoryLoader(DATA_YOLU, glob="*.pdf", loader_cls=PyPDFLoader)
            docs = loader.load()

            # Parçala
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Kaydet
            embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=DB_YOLU)
    else:
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        vectorstore = Chroma(persist_directory=DB_YOLU, embedding_function=embedding_model)

    # RETRIEVER (Arayıcı) OLUŞTUR
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # KRİTİK ADIM: RETRIEVER'I TOOL'A ÇEVİRME
    # Artık bu bir "Fonksiyon" oldu. Ajan bunu gerektiğinde çağıracak.
    retriever_tool = create_retriever_tool(
        retriever,
        name="ilac_bilgi_kaynagi",
        description="İlaçların kullanımı, yan etkileri, dozajları ve içerikleri hakkında bilgi aramak için bu aracı kullan. Soruyu olduğu gibi veya anahtar kelimelerle gönder."
    )

    return [retriever_tool]


# 3. AJANI (AGENT) KURMA
def initialize_agent(tools):
    # LLM (Beyin)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # SYSTEM PROMPT (Ajanın Anayasası) [cite: 26]
    # ReAct döngüsünü (Thought-Action-Observation) öğretiyoruz.
    prompt_template = """
    Sen uzman bir Eczacı Asistanısın. Görevin kullanıcının ilaçlarla ilgili sorularını yanıtlamak.

    Kullanabileceğin araçlar:
    {tools}

    Soruya cevap verirken şu formatı KESİNLİKLE takip et (ReAct Mimarisi):

    Soru: Kullanıcının girdiği soru
    Thought: (Düşünce) Soruyu cevaplamak için ne yapmalıyım? Hangi aracı kullanmalıyım? 
    Action: (Eylem) [{tool_names}] listesinden bir araç seç.
    Action Input: (Eylem Girdisi) Araç için gerekli arama kelimesi.
    Observation: (Gözlem) Araçtan gelen sonuç.
    ... (Gerekirse Düşünce/Eylem/Gözlem adımlarını tekrarla)
    Thought: Artık cevabı biliyorum.
    Final Answer: (Nihai Cevap) Kullanıcıya vereceğin Türkçe ve açıklayıcı cevap.

    KURALLAR:
    1. Asla kendi bilgine güvenme, MUTLAKA "ilac_bilgi_kaynagi" aracını kullan.
    2. Eğer hesaplama gerekirse (örn: kilo başına dozaj), adım adım hesapla.
    3. Cevapların Türkçe olsun.

    Başlayalım!

    Soru: {input}
    Thought:{agent_scratchpad}
    """

    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(prompt_template)

    # Ajanı Oluştur
    agent = create_react_agent(llm, tools, prompt)

    # Executor (Çalıştırıcı) - Sonsuz döngü koruması [cite: 52]
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Arka planda nasıl düşündüğünü terminale yazar (Trace)
        handle_parsing_errors=True,
        max_iterations=5  # Sonsuz döngüye girmesin diye limit
    )

    return agent_executor


# 4. ARAYÜZ (FRONTEND)
try:
    tools = setup_knowledge_base()
    agent_executor = initialize_agent(tools)
except Exception as e:
    st.error(f"Sistem başlatılamadı: {e}")
    st.stop()

# Sohbet Geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Soru Cevap Döngüsü
if user_input := st.chat_input(
        "İlaç hakkında ne öğrenmek istiyorsun? (Örn: Çocuğum 20kg, X şurubundan ne kadar vermeliyim?)"):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Eczacı asistanı prospektüsleri inceliyor..."):
            try:
                # Ajanı çalıştır
                response = agent_executor.invoke({"input": user_input})
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")