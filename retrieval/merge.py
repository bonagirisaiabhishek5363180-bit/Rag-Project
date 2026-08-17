from retrieval.vector_search import vector_search
from retrieval.bm25_search import bm25_search
def merge_results(query):
    vector_results=vector_search(query)
    bm25_results=bm25_search(query)
    combined_results=[]
    seen=set()
    for doc in vector_results:
        text=doc.page_content
        if text not in seen:
            combined_results.append(doc)
            seen.add(text)
    for doc in bm25_results:
        text=doc.page_content
        if text not in seen:
            combined_results.append(doc)
            seen.add(text)
    return combined_results