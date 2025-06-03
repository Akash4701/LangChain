from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Load environment variables
load_dotenv()

# List of documents
documents = [
    "Virat Kohli is known for his aggressive batting style and is one of the highest run-scorers in international cricket.",
    "MS Dhoni, also called 'Captain Cool,' is famous for leading India to victory in all major ICC tournaments.",
    "Rohit Sharma holds the record for the highest individual score in ODI cricket, with 264 runs.",
    "Shreyas Iyer is a stylish middle-order batsman who has played crucial innings for India in both ODIs and T20s.",
    "Dinesh Karthik is a versatile wicketkeeper-batsman known for his finishing abilities and comeback performances."
]

# Your query
query = "Tell about mahendra singh dhoni"

# Initialize embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001" 
)

# Get document embeddings
doc_embeddings = embedding.embed_documents(documents)

# Get query embedding
query_embedding = embedding.embed_query(query)
 
# print("Query Embedding:", query_embedding)

similarity=cosine_similarity([query_embedding], doc_embeddings)[0]

print("Cosine Similarity:", similarity)
index,value=sorted(list(enumerate(similarity)),key=lambda x:x[1])[-1]

print('documents',documents[index])
print(query)