from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
vector_db=Chroma(persist_directory="./chroma_db",embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever=vector_db.as_retriever(
    search_type="mmr",search_kwargs={"k":3,"fetch_k":20}
    )
def vector_search(query):
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs