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

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}
structured_output=model.with_structured_output(json_schema)

result=structured_output.invoke("""Absolutely impressed with the Anker Soundcore! For such a compact and affordable speaker, the sound quality is surprisingly rich and clear, with decent bass. Battery life is a big win — it easily lasts over 12 hours on a single charge. It’s lightweight, easy to carry, and perfect for travel or casual home use. Highly recommended for anyone looking for a budget-friendly yet powerful portable speaker.""")

print(result)


    
    
    


