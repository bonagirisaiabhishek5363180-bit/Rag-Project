import json
from retrieval.reranker import Reranker
from retrieval.merge import merge_results   
from dotenv import load_dotenv
from google import genai
import yaml
from groq import Groq
import os

with open("prompts/qa_2_basic_prompt.yaml",encoding="utf-8") as f:
    prompt_template=yaml.safe_load(f)
prompt_data=prompt_template["prompt"]
load_dotenv()
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
reranker=Reranker()
def rag_run(query):
    retrived_docs=merge_results(query)
    retrived_docs=reranker(query,retrived_docs,top_k=3)
    context="\n".join(doc.page_content for doc in retrived_docs)
    prompt=prompt_data.format(context=context,query=query)
    try:
        # Primary LLM
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        answer = response.text
        model_used = "gemini-3.8-flash"

    except Exception as e:
        # Fallback LLM
        status_code = getattr(e, "code", None)

        if status_code in (401, 429, 503):

            print(
                f"Gemini failed with {status_code}. "
                "Switching to Groq..."
            )

            response = groq_client.chat.completions.create(
                model="gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content
            model_used = "groq/meta-llama/llama-3.3-70b"

        else:
            raise
    return {
            "question": query,
            "answer": answer,
            "model": model_used,
            "contexts": [
        doc.page_content
        for doc in retrived_docs
        ],
            "sources":[doc.metadata.get("source", "Unknown") for doc in retrived_docs]
            }