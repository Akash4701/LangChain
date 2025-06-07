from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

prompt=PromptTemplate(
    template="You are a helpful assistant. Answer the question: {question}",
    input_variables=["question"]
)

parsers=StrOutputParser()

chain= prompt|model|parsers

result=chain.invoke({"question":"What is the capital of France?"})

print(result)

chain.get_graph().print_ascii()