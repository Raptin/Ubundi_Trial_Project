import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from promptBuilder import get_response, PROMPT_MODIFIERS

load_dotenv()


st.title("Chat with your CV")


selected_mode = st.selectbox(
    "Select Chat Mode:",
    options=list(PROMPT_MODIFIERS.keys()),
    index=0,
    help="Choose how you want the AI to respond",
)

if selected_mode != "Default":
    st.info(f"🎭 **{selected_mode} Mode Active**")


if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# user input
if prompt := st.chat_input("Say something"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    result = get_response(prompt, selected_mode)
    st.session_state.messages.append({"role": "assistant", "content": result.content})
    with st.chat_message("assistant"):
        st.write(result.content)
