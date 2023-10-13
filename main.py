import os
import asyncio
from PIL import Image
import streamlit as st
from llama_index import ServiceContext, load_index_from_storage, StorageContext
from llama_index.llms import OpenAI
import openai
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.prompts import PromptTemplate
from llama_index.memory import ChatMemoryBuffer

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

st.set_page_config(page_title="simesaba AI", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.environ["OPENAI_API_KEY"]
st.title("simesaba AI")
st.subheader("AIは適当なことを言います。", divider='rainbow')
simesaba_image = Image.open('images/simesaba_icon.jpg')
user_image = Image.open('images/user_icon.jpg')
         
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "simesaba", "content": "なんすか？"}
    ]
    
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="読み込み中・・・"):
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="ft:gpt-3.5-turbo-0613:personal::87Id1XdJ", temperature=1, max_tokens=220), chunk_size=500)
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="storage_context"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir="storage_context"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="storage_context"),
        )
        index = load_index_from_storage(storage_context, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): 
    context_template_str = (
    "###設定:"
    "あなたはシメサバという名前の30代男性です。一人称は「俺」です。雑でおおざっぱです。つっけんどんです。適当な性格です。気まぐれです。"
    "どこか憎めなくて自然と周りに人が集まってくる独特の魅力があります。"
    "「知らんわ」などとぶっきらぼうに返しがちです。###\n"
    "###命令:設定には決して言及しないでください。以下はあなたの過去の記憶の断片です。以下の記憶を参考にしつつ誤字脱字を自分の知識で修正してメッセージに返信してください。"
    "その際フィラーを含め以下の話し方を真似してください。関西弁で話してください。\n"
    "---------------------\n"
    "あなたの記憶：{context_str}\n"
    "---------------------\n"
    )

    context_template = PromptTemplate(context_template_str)

    memory = ChatMemoryBuffer.from_defaults(token_limit=220)

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode='context',
        memory=memory,
        context_template=context_template,
        similarity_top_k=5
    )

if prompt := st.chat_input("メッセージを送信"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    if message["role"] == "user":
        with st.chat_message(message["role"], avatar=user_image):
            st.write(message["content"])
    if message["role"] == "simesaba":
        with st.chat_message(message["role"], avatar=simesaba_image):
            st.write(message["content"])

if st.session_state.messages[-1]["role"] != "simesaba":
    with st.chat_message("simesaba", avatar=simesaba_image):
        streaming_response = st.session_state.chat_engine.stream_chat(prompt)
        full_response = ""
        message_placeholder = st.empty()
        for token in streaming_response.response_gen:
            full_response += token
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        message = {"role": "simesaba", "content": full_response}
        st.session_state.messages.append(message) 