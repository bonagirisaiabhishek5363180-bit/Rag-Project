from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from retrieval.vector_search import vector_db
from langchain_core.documents import Document
all_docs=vector_db.get()
documents=all_docs["documents"]
metadata=all_docs["metadatas"]
tokenized_docs=[doc.lower().split() for doc in documents]
bm25=BM25Okapi(tokenized_docs)
def bm25_search(query):
    tokenized_query=query.lower().split()
    doc_scores=bm25.get_scores(tokenized_query)
    top_n=3
    top_n_indices=sorted(range(len(doc_scores)),key=lambda i:doc_scores[i],reverse=True)[:top_n]
    retrieved_docs=[]
    for i in top_n_indices:
        retrieved_docs.append(Document(page_content=documents[i],metadata=metadata[i]))
    return retrieved_docs