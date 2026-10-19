from langchain_community.document_loaders import PyPDFLoader       
from langchain_text_splitters import RecursiveCharacterTextSplitter

obj=PyPDFLoader('source/fluid.pdf')
docs=obj.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)

text=splitter.split_documents(docs)
print(len(text))