import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

ragas_llm = LangchainLLMWrapper(llm)

print("Gemini Ragas evaluator created successfully")
print(ragas_llm)