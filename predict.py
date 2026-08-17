from langchain_chroma import Chroma
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
import os
from retrieval.merge import merge_results
from dotenv import load_dotenv
import yaml
with open("prompts/qa_1_basic_prompt.yaml","r",encoding="utf-8") as f:
    prompt_data=yaml.safe_load(f)
system_template = prompt_data["system"]
user_template = prompt_data["user"]
load_dotenv(dotenv_path=".env")
apikey=os.getenv("GEMINI_API_KEY")
while True:
    input_query=input("Enter your query (or 'exit' to quit): ").strip()
    if input_query.lower() == 'exit':
        break
    retrived_docs=merge_results(input_query)
    client=genai.Client(api_key=apikey)
    context="\n".join([doc.page_content for doc in retrived_docs])
    prompt = f"{system_template}\n\n{user_template.format(context=context, input_query=input_query)}"
    response=client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    print("Answer:")
    print(response.text)
    for i,doc in enumerate(retrived_docs):
        print(f"Source [{i+1}]: {doc.metadata.get('source','Unknown Source')}")
