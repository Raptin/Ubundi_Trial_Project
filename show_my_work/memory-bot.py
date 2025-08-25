import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore


warnings.filterwarnings("ignore")

load_dotenv()

chat_history = []

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=vectorstore.as_retriever()
    )

    result = qa.invoke(
        {"question": "What skills does this person have?", "chat_history": []}
    )
    print(result)

    history = (result["question"], result["answer"])
    chat_history.append(history)

    result = qa.invoke(
        {
            "question": "Would these skills be useful for a software engineer?",
            "chat_history": chat_history,
        }
    )
    print(result)
