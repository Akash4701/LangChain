from typing import Literal, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
   
    timeout=None,
    max_retries=2,
)

class Review(BaseModel):
    key_themes:list[str]=Field(description="Key themes or topics discussed in the review")
    sentiment:Literal["pos","neg"]=Field(description="Sentiment of the review, True for positive, False for negative")
    summary:str=Field(description="A brief summary of the review")
    pros:Optional[list[str]]=Field(default=None,description="List of positive aspects mentioned in the review")
    cons:Optional[list[str]]=Field(default=None,description="List of negative aspects mentioned in the review")
    name:Optional[str]=Field(default=None,description="Name of the reviewer")
structured_output=model.with_structured_output(Review)

result=structured_output.invoke("""Absolutely impressed with the Anker Soundcore! For such a compact and affordable speaker, the sound quality is surprisingly rich and clear, with decent bass. Battery life is a big win — it easily lasts over 12 hours on a single charge. It’s lightweight, easy to carry, and perfect for travel or casual home use. Highly recommended for anyone looking for a budget-friendly yet powerful portable speaker.""")

print(result)


    
    
    


