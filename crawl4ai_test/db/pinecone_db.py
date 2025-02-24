# Import the Pinecone library
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Set
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
import logging

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

logging.basicConfig(level=logging.DEBUG)

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=api_key)
model = SentenceTransformer(model_name)

def create_index_if_not_exists(index_name: str, dimension: int):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
        pc.create_index(index_name, spec, dimension=dimension)

def store_embeddings_in_pinecone(index_name: str, data: Dict[str, str]):
    dimension = 384
    create_index_if_not_exists(index_name, dimension=dimension)
    index = pc.Index(index_name)
    vectors = []
    for url, content in data.items():
        embedding = model.encode(content).tolist()
        # Ensure embedding is in the correct format
        embedding = [float(x) for x in embedding]
        if len(embedding) != dimension:
            logging.error(f"Embedding dimension mismatch for {url}: expected {dimension}, got {len(embedding)}")
            continue
        logging.debug(f"Storing embedding for {url}: {embedding}")
        vectors.append({"id": url, "values": embedding})
    index.upsert(vectors)

def search_pinecone(index_name: str, query: str, top_k: int = 10):
    dimension = 384
    index = pc.Index(index_name)
    query_embedding = model.encode(query).tolist()
    # Ensure query_embedding is in the correct format
    query_embedding = [float(x) for x in query_embedding]
    if len(query_embedding) != dimension:
        logging.error(f"Query embedding dimension mismatch: expected {dimension}, got {len(query_embedding)}")
        return []
    logging.debug(f"Query embedding: {query_embedding}")
    results = index.query(vector=query_embedding, top_k=top_k,include_values=True,include_metadata=True)
    return results