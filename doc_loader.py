import os
from google import genai
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.csv':
        loader = CSVLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
    elif file_extension in ['.docx', '.doc']:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_extension in ['.pptx', '.ppt']:
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"does not support file type:{file_extension}")
    return loader.load()

alldocs=[]
while True:
    file_path=input("enter the file path (or 'done' to finish): ").strip().strip('"')
    if file_path == 'done':
        break
    else:
        try:
            documents=load_document(file_path)
            for doc in documents:
                print(f"doc metadata: {doc.metadata}")
                print(f"doc page content: {doc.page_content[:300]}")
            alldocs.extend(documents)
        except Exception as e:
            print(f"Error loading document: {e}")
print(f"Total documents loaded: {len(alldocs)}")
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db=Chroma(persist_directory="./chroma_db",embedding_function=embeddings)
if len(alldocs) > 0:
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150,separators=["\n\n","\n",",","."," ",""])
    chunks=text_splitter.split_documents(alldocs)
    vector_db.add_documents(chunks)
    DEBUG=False
    if DEBUG:
        print(f"Total chunks created: {len(chunks)}")
        for i,chunk in enumerate(chunks[:5]):
            print(f"Chunk {i+1}:")
            print("Page Content:")
            print(chunk.page_content[:300])
            print("Metadata:")
            print(chunk.metadata)
            print("-"*50)

    print("Documents added to Chroma vector database.")
    print("number of documents in the vector database:",vector_db._collection.count())
else:
    print("No new documents uploaded.")
    print("Using existing database.")
print("Number of documents in DB:",vector_db._collection.count())

    