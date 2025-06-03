import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # Loads environment variables from .env file

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # If you want to confirm the key loaded
# print("GEMINI_API_KEY Loaded:", bool(GEMINI_API_KEY))

# Note: There is no 'Langchain' class that takes api_key in LangChain
# If you're trying to verify API access, use a call to ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    ("tELL ME,WHAT'S MY NAME"),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
