from sentence_transformers  import CrossEncoder
class Reranker:
    def __init__(self,model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name=model_name
        self.model=CrossEncoder(self.model_name)
    def __call__(self,input_query,retrived_docs,top_k=3):
        self.input_query=input_query
        self.retrived_docs=retrived_docs
        self.top_k=top_k
        pairs=[(self.input_query,doc.page_content) for doc in self.retrived_docs]
        scores=self.model.predict(pairs)
        ranked_docs=sorted(zip(self.retrived_docs,scores),key=lambda x:x[1],reverse=True)
        top_k_docs=[doc for doc,score in ranked_docs[:self.top_k]]
        return top_k_docs