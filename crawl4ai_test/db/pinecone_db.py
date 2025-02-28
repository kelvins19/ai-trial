# Import the Pinecone library
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Set
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from langchain.text_splitter import MarkdownTextSplitter

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


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

def store_embeddings_in_pinecone(index_name: str, data: Dict[str, str], chunk_size: int = 100):
    print(f"Storing {len(data)} embeddings in Pinecone index {index_name}")
    dimension = 384
    create_index_if_not_exists(index_name, dimension=dimension)
    index = pc.Index(index_name)
    vectors = []
    chunk_overlap = min(chunk_size // 2, 50)  # Ensure chunk_overlap is smaller than chunk_size
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for url, content in data.items():
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            # Ensure embedding is in the correct format
            embedding = [float(x) for x in embedding]
            if len(embedding) != dimension:
                print(f"Embedding dimension mismatch for {url}: expected {dimension}, got {len(embedding)}")
                continue
            vectors.append({"id": f"{url}_chunk_{i}", "values": embedding})
            if len(vectors) >= chunk_size:
                index.upsert(vectors)
                vectors = []
    if vectors:
        index.upsert(vectors)

def search_pinecone(index_name: str, query: str, top_k: int = 10):
    dimension = 384
    index = pc.Index(index_name)
    query_embedding = model.encode(query).tolist()
    # Ensure query_embedding is in the correct format
    query_embedding = [float(x) for x in query_embedding]
    if len(query_embedding) != dimension:
        print(f"Query embedding dimension mismatch: expected {dimension}, got {len(query_embedding)}")
        return []
    # print(f"Query embedding: {query_embedding}")
    results = index.query(vector=query_embedding, top_k=top_k,include_values=True,include_metadata=True)
    return results