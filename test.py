import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

print("Groq key loaded:", os.getenv("GROQ_API_KEY") is not None)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
try:
    response = client.chat.completions.create(
        model="groq/meta-llama/llama-3.3-70b",
        messages=[
            {
                "role": "user",
                "content": "Say hello in one sentence."
            }
        ]
    )
except Exception as e:
    print("Error occurred while creating Groq client:", str(e))
    raise
print(response.choices[0].message.content)