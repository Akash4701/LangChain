from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage



load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

chathistory=[
    SystemMessage(content="You are my Ai girlfriend")
]

while True:
    user_input=input("YOU:")
    chathistory.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    result=model.invoke(chathistory)
    print("AI:",result.content)
    chathistory.append(AIMessage(content=result.content))
print("Chat History:",chathistory)   
    
    
    