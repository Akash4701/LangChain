from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
   
    timeout=None,
    max_retries=2,
)

class Review(TypedDict):
    sentiment: bool
    summary:str
    rating:int
    
structured_output=model.with_structured_output(Review)

result=structured_output.invoke("""Absolutely impressed with the Anker Soundcore! For such a compact and affordable speaker, the sound quality is surprisingly rich and clear, with decent bass. Battery life is a big win — it easily lasts over 12 hours on a single charge. It’s lightweight, easy to carry, and perfect for travel or casual home use. Highly recommended for anyone looking for a budget-friendly yet powerful portable speaker.""")

print(result)


    
    
    


