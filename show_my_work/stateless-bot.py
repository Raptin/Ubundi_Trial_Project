import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)

qa = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

result = qa.invoke("Who's CV is this? And what do they do?")
print(result)

# result = qa.invoke("What coding languages do you know?")
# print(result["result"])
