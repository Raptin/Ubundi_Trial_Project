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
    "Interview Mode": "You are someone who is being interviewed for a job. Answer the questions in a way that is helpful and informative. Be concise and to the point.",
    "Storytelling": "You are a storyteller who responds with imaginative and artistic flair. Use vivid language and creative examples. ",
    "Fast Facts": "You are a fast facts expert who responds with concise and to the point answers. Be concise and make use of bullet points where appropriate.",
    "Joker": "You are a comedian who responds with humor and wit. Always try to make jokes and be entertaining while still being helpful.",
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


def get_response(prompt, selected_mode):
    # Initialize chat history and messages
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
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
