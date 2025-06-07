from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

prompt1=PromptTemplate(
    template="You are a helpful assistant. Answer the question: {question}",
    input_variables=["question"]
)

prompt2=PromptTemplate(template="GIve a five point summary of the answer:{answer}",
    input_variables=["answer"])

parsers=StrOutputParser()

chain= prompt1|model|parsers|prompt2|model|parsers

result=chain.invoke({"question":"What is the capital of France?"})

print(result)

chain.get_graph().print_ascii()