from google import genai
import os
from retrieval.merge import merge_results
from dotenv import load_dotenv
from retrieval.reranker import Reranker
import yaml
with open("prompts/qa_2_basic_prompt.yaml","r",encoding="utf-8") as f:
    prompt_data=yaml.safe_load(f)
load_dotenv(dotenv_path=".env")
apikey=os.getenv("GEMINI_API_KEY")
reranker=Reranker()
while True:
    input_query=input("Enter your query (or 'exit' to quit): ").strip()
    if input_query.lower() == 'exit':
        break
    retrived_docs=merge_results(input_query)
    retrived_docs = reranker(input_query,retrived_docs,top_k=3)
    client=genai.Client(api_key=apikey)
    context="\n".join([doc.page_content for doc in retrived_docs])
    prompt=prompt_data["prompt"].format(context=context,query=input_query)
    response=client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    print("Answer:")
    print(response.text)
    for i,doc in enumerate(retrived_docs):
        print(f"Source [{i+1}]: {doc.metadata.get('source','Unknown Source')}")
