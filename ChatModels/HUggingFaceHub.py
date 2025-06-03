from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# load_dotenv()

print(load_dotenv())

llm= HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
print("HuggingFace LLM Loaded")
model= ChatHuggingFace(
    llm=llm
)
print("HuggingFace LLM Loaded")

msg=llm.invoke("what is the capital of India")
print("Message invoked")

print(msg)