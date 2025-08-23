import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
vectorstore = PineconeVectorStore(
    index_name=st.secrets["INDEX_NAME"], embedding=embeddings
)


# initialize chat and qa
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)
retriever = vectorstore.as_retriever()

PROMPT_MODIFIERS = {
    "Default": "Do not just repeat the context. Be helpful and informative.",
    "Joker": "You are a comedian who responds with humor and wit. Always try to make jokes and be entertaining while still being helpful.",
    "Formal": "You are a professional assistant providing formal and concise responses. Use professional language and maintain a serious tone.",
    "Casual": "You are a friendly assistant responding in a relaxed and conversational tone. Be warm and approachable. ",
    "Academic": "You are an academic expert providing detailed, scholarly responses with proper citations and formal language. ",
    "Creative": "You are a creative writer who responds with imaginative and artistic flair. Use vivid language and creative examples. ",
}

prompt_template = PromptTemplate(
    input_variables=["question", "chat_history", "mode_instructions", "context"],
    template="""
    You are representing a digital version of a person.
    You are given context about the person from documents about them.
    You are given a question and a chat history.
    You are also given a mode instructions.
    You need to answer the question based on the chat history, mode instructions, and context.
    Chat history:
    {chat_history}
    Mode instructions:
    {mode_instructions}
    Context:
    {context}
    Question:
    {question}
    Answer:
    """,
)


# Initialize chat history and messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_response(prompt, selected_mode):
    # First we get the relevant data from the vector store
    docs = retriever.invoke(prompt)
    context = "\n".join([doc.page_content for doc in docs])

    # Then we use the chat model to generate a response
    response = chat.invoke(
        prompt_template.format(
            question=prompt,
            chat_history=st.session_state.chat_history,
            mode_instructions=PROMPT_MODIFIERS[selected_mode],
            context=context,
        )
    )
    st.session_state.chat_history.append((prompt, response.content))

    return response
