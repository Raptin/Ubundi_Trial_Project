import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from promptBuilder import get_response, PROMPT_MODIFIERS
from embedding.ingestion import process_uploaded_file

load_dotenv()


st.title("Ruan's AI")


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

# File upload functionality in sidebar
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document to chat about",
        type=["pdf", "txt", "csv", "doc", "docx"],
        help="Upload a document to add to the knowledge base for this conversation",
    )

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    # Save uploaded file temporarily
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the uploaded file (you can expand this based on your needs)
    with st.spinner("Processing uploaded document..."):
        try:
            # Here you could integrate with your document loading functionality
            process_uploaded_file(temp_file_path)
            # from ingestion.py if needed for real-time processing
            st.info(
                "Document uploaded successfully! You can now ask questions about it."
            )

            # Clean up temp file
            os.remove(temp_file_path)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
