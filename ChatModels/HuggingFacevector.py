from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "This is a test sentence for embedding."
documents=(
    "This is a test sentence for embedding.",
    "Another sentence to test the embeddings.",
    "Yet another example to see how embeddings work."
    
)
# embeddings=embeddings.embed_query(text)
embeddings=embeddings.embed_documents(documents)
print("Embeddings generated:", embeddings)
