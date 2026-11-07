import os
from typing import List, Tuple

import yaml
from dotenv import load_dotenv
from google import genai
from groq import Groq

from retrieval.merge import merge_results
from retrieval.reranker import Reranker

PIPELINE_STAGES = [
    "document ingestion",
    "document cleaning and preprocessing",
    "chunking and indexing",
    "vector + BM25 retrieval",
    "reranking",
    "prompt construction",
    "LLM answer generation",
    "result packaging and source tracing",
]


def load_prompt_template(path: str = "prompts/qa_2_basic_prompt.yaml") -> str:
    with open(path, "r", encoding="utf-8") as file:
        prompt_template = yaml.safe_load(file)
    return prompt_template["prompt"]


PROMPT_TEMPLATE = load_prompt_template()
load_dotenv()


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


def retrieve_documents(query: str):
    return merge_results(query)


def rerank_documents(query: str, docs, top_k: int = 3):
    reranker = Reranker()
    return reranker(query, docs, top_k=top_k)


def build_context(documents) -> str:
    return "\n".join(doc.page_content for doc in documents)


def build_prompt(query: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, query=query)


def generate_with_gemini(prompt: str, model_name: str = None):
    client = get_gemini_client()
    model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text, model_name


def generate_with_groq(prompt: str, model_name: str = None):
    client = get_groq_client()
    model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content, model_name


def generate_answer(prompt: str) -> Tuple[str, str]:
    try:
        return generate_with_gemini(prompt)
    except Exception as exc:
        status_code = getattr(exc, "code", None)
        if status_code in (401, 429, 503):
            print(f"Gemini failed with {status_code}. Switching to Groq...")
            return generate_with_groq(prompt)
        raise


def rag_run(query: str):
    if not query or not str(query).strip():
        raise ValueError("Query cannot be empty.")

    normalized_query = str(query).strip()

    retrieved_docs = retrieve_documents(normalized_query)
    ranked_docs = rerank_documents(normalized_query, retrieved_docs, top_k=3)
    context = build_context(ranked_docs)
    prompt = build_prompt(normalized_query, context)
    answer, model_used = generate_answer(prompt)

    return {
        "question": normalized_query,
        "answer": answer,
        "model": model_used,
        "contexts": [doc.page_content for doc in ranked_docs],
        "sources": [doc.metadata.get("source", "Unknown") for doc in ranked_docs],
        "pipeline_stages": PIPELINE_STAGES,
    }
