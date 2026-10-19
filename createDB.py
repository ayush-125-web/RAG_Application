from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunkGen.chunk import text
from dotenv import load_dotenv

load_dotenv()


embeddings=HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")


vector_store = Chroma.from_documents(
    documents=text,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

