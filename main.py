from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
model=ChatGoogleGenerativeAI(model='gemini-3.7-flash',temperature=0.3)

prompt=ChatPromptTemplate.from_messages(
    [
        (
            'system',"""
             You are a English Teacher who gives answer from the context given to you which 
             is {content} in a very professional tone
        """),
        (
            'human','{query}'
        )

    ]
)


## loads the VectorDB
vector_store=Chroma(
    collection_name='pdf_chunks',
    embedding_function=embeddings,
    persist_directory='chroma_langchain_eng11_db'
)

## print("Total chunks Loaded:", vector_store._collection.count())


query=str(input("Enter Your Question : "))

# results = vector_store.similarity_search(
#     query,
#     k=8
# )

retreiver=vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k':8
    }
)
# for doc in results:
#     print(doc.page_content)
#     print('\n\n')

results=retreiver.invoke(query)

final_prompt=prompt.invoke(
    {
        'content':results,
        'query':query
    }
)

response=model.invoke(final_prompt)
print(response.text)

##print(len(results))
