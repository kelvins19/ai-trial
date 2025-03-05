# Import the Pinecone library
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Set
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveJsonSplitter
from .db import upsert_docstore_in_db
import json

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")


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

def store_embeddings_in_pinecone(index_name: str, data: Dict[str, str], model_name: str):
    print(f"Storing embeddings in Pinecone index {index_name}")
    dimension = 768
    create_index_if_not_exists(index_name, dimension=dimension)
    index = pc.Index(index_name)
    vectors = []

    content = json.dumps(data)
    embedding = model.encode(content).tolist()
    # Ensure embedding is in the correct format
    embedding = [float(x) for x in embedding]
    if len(embedding) != dimension:
        print(f"Embedding dimension mismatch: expected {dimension}, got {len(embedding)}")
        return

    doc_id = f"{model_name}_data"
    vectors.append({"id": doc_id, "values": embedding, "metadata": {"doc_id": doc_id, "doc_type": "text", "filename": doc_id, "summary": content}})
    # Upsert plaintext content to local DB
    upsert_docstore_in_db(doc_id, content)

    if vectors:
        index.upsert(vectors)
