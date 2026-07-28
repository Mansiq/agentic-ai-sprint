from dotenv import load_dotenv
from groq import Groq
import os
from google import genai

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY")

prompt = "Explain recursion in one sentence."
## Groq's API calling
def call_groq(prompt):
    client = Groq(api_key = groq_key)
    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [
            {
                "role":"user",
                "content":prompt
            }
        ]
    )
    return response.choices[0].message.content

## Gemini's API calling
def call_gemini(prompt):
    client1 = genai.Client(api_key = gemini_key)
    response = client1.models.generate_content(
    model = "gemini-3.1-flash-lite",
    contents = prompt)
    return response.text

groq_call = call_groq(prompt)
gemini_answer = call_gemini(prompt)
print("Groq reaponse :",groq_call)
print("Gemini response :",gemini_answer)



