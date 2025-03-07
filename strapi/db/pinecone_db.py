# Import the Pinecone library
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import Dict
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from langchain.text_splitter import MarkdownTextSplitter, RecursiveJsonSplitter, RecursiveCharacterTextSplitter
from .db import upsert_docstore_in_db
import json
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import numpy as np
from strapi_rag import KeywordGenerator, SummaryGenerator

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
embed_access_key = "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"

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=api_key)

# Initialize the HuggingFaceInferenceAPIEmbeddings with your model name
model = HuggingFaceInferenceAPIEmbeddings(model_name=model_name, api_key=embed_access_key)

def create_index_if_not_exists(index_name: str, dimension: int):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
        pc.create_index(index_name, spec, dimension=dimension)

def format_json_to_markdown(json_data) -> str:
    def format_dict(d, indent=0):
        md = ""
        for key, value in d.items():
            md += ' ' * indent + f"- **{key}**: "
            if isinstance(value, dict):
                md += "\n" + format_dict(value, indent + 2)
            elif isinstance(value, list):
                md += "\n" + format_list(value, indent + 2)
            else:
                md += f"{value}\n"
        return md

    def format_list(lst, indent=0):
        md = ""
        for item in lst:
            if isinstance(item, dict):
                md += format_dict(item, indent)
            else:
                md += ' ' * indent + f"- {item}\n"
        return md

    if isinstance(json_data, dict):
        return format_dict(json_data)
    elif isinstance(json_data, list):
        return format_list(json_data)
    else:
        return str(json_data)

def chunk_json_data(json_data, max_metadata_size):
    splitter = RecursiveJsonSplitter(min_chunk_size=10, max_chunk_size=max_metadata_size)
    # print(f"Json dumps {json.dumps(json_data)}")
    chunks = splitter.split_text(json_data, True, True)
    return chunks

def chunk_json_data_text(json_data):
    if isinstance(json_data, dict):
        json_data = json.dumps(json_data)
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(json_data)
    return chunks

def store_embeddings_in_pinecone(index_name: str, data: Dict[str, str], model_name: str):
    print(f"Storing {model_name} embeddings in Pinecone index {index_name}")
    dimension = 768
    create_index_if_not_exists(index_name, dimension=dimension)
    index = pc.Index(index_name)
    vectors = []
    max_metadata_size = 1024  

    content = format_json_to_markdown(data['data'])
    if len(content.encode('utf-8')) > max_metadata_size:
        chunks = chunk_json_data(data['data'], max_metadata_size)
    else:
        chunks = [content]

    for i, chunk in enumerate(chunks):
        try:
            if len(chunk.encode('utf-8')) > max_metadata_size:
                print(f"Chunk size exceeds the limit: {len(chunk.encode('utf-8'))} bytes")
                continue

            embedding = model.embed_query(chunk)
            embedding = [float(x) for x in embedding]

            if len(embedding) != dimension:
                print(f"Embedding dimension mismatch: expected {dimension}, got {len(embedding)}")
                continue

            doc_id = f"{model_name}_chunk_{i}"
            vectors.append({"id": doc_id, "values": embedding, "metadata": {"doc_id": doc_id, "doc_type": "text", "filename": doc_id, "summary": chunk, "page_number": 1}})
            # Upsert plaintext content to local DB
            upsert_docstore_in_db(doc_id, chunk)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    if vectors:
        index.upsert(vectors)

async def store_embeddings_in_pinecone_chunkjson(index_name: str, data: Dict[str, str], model_name: str):
    print(f"Storing {model_name} embeddings in Pinecone index {index_name}")
    dimension = 768
    create_index_if_not_exists(index_name, dimension=dimension)
    index = pc.Index(index_name)
    vectors = []
    max_metadata_size = 1024  

    # Initialize the KeywordGenerator
    keyword_generator = KeywordGenerator(
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        prompt=""
    )

    # Initialize the SummaryGenerator
    sumamry_generator = SummaryGenerator(
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        prompt=""
    )

    # Ensure data is a JSON string
    json_data = json.dumps(data)
    chunks = chunk_json_data_text(json_data)

    for i, chunk in enumerate(chunks):
        try:
            embedding = model.embed_query(chunk)
            embedding = [float(x) for x in embedding]

            if len(chunk.encode('utf-8')) > 40960:
                print(f"Chunk {chunk}")
                print(f"Chunk size exceeds the limit: {len(chunk.encode('utf-8'))} bytes")
                continue

            if len(embedding) != dimension:
                print(f"Embedding dimension mismatch: expected {dimension}, got {len(embedding)}")
                continue

            doc_id = f"{model_name}_chunk_{i}_json"

            # Generate keywords for the chunk
            keyword_response = await keyword_generator.generate_keywords(input_text=chunk, phone_number="1234567890")
            keywords = json.loads(keyword_response).get("keywords", [])

            summary_response = await sumamry_generator.generate_summary(input_text=chunk, phone_number="123456789")

            vectors.append({
                "id": doc_id, 
                "values": embedding, 
                "metadata": {
                    "doc_id": doc_id, 
                    "doc_type": "text", 
                    "filename": doc_id, 
                    "summary": summary_response, 
                    "page_number": 1,
                    "keywords": keywords
                }
            })
            # Upsert plaintext content to local DB
            upsert_docstore_in_db(doc_id, chunk)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    if vectors:
        index.upsert(vectors)