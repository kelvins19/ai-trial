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
from constants import STRAPI_KEYWORD_GENERATOR_PROMPT_1, STRAPI_SUMMARY_GENERATOR_PROMPT
from strapi_keyword_summary_generator import query_formatter
import datetime

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
        # pc.create_index(index_name, spec, vector_type=VectorType.SPARSE, metric=Metric.DOTPRODUCT)
        pc.create_index_for_model(index_name, 
                        cloud="aws",
                        region="us-east-1", 
                        embed={
                            "model":"pinecone-sparse-english-v0",
                            "field_map":{"text": "chunk_text"}
                        },
                        )

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
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_text(json_data)
    return chunks

def search_data_in_pinecone_sparse(index_name: str, query: str, k: int =20):
    start_time = datetime.datetime.now()
    embeddings = pc.inference.embed(
                model="pinecone-sparse-english-v0",
                inputs=query,
                parameters={"input_type": "query"}
            )
    # print(f"Embeddings {embeddings}")
    sparse_values = embeddings.data[0]['sparse_values']
    sparse_indices = embeddings.data[0]['sparse_indices']
    end_time = datetime.datetime.now()
    print(f"Time taken for convert to vector: {end_time - start_time} seconds")

    start_time = datetime.datetime.now()
    index = pc.Index(index_name)
    retrieved_docs = index.query(
            namespace="",
            sparse_vector={
                "values": sparse_values,
                "indices": sparse_indices
            },
            top_k=k,
            include_metadata=True,
            include_values=False
        )
    
    end_time = datetime.datetime.now()
    print(f"Time taken for retrieval from pinecone: {end_time - start_time} seconds")
    return retrieved_docs

def search_data_in_pinecone(index_name: str, query: str, k: int = 20): 
    index = pc.Index(index_name)
    retrieved_docs = index.search_records(
            namespace="", 
            query={
                "inputs": {"text": query}, 
                "top_k": k
            },
            fields=["summary", "keywords", "chunk_text"]
        )
    
    return retrieved_docs

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

def escape_curly_brackets(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")

async def store_embeddings_in_pinecone_chunkjson(index_name: str, data: Dict[str, str], model_name: str):
    print(f"Storing {model_name} embeddings in Pinecone index {index_name}")
    dimension = 1024
    create_index_if_not_exists(index_name, dimension=dimension)
    index = pc.Index(index_name)
    vectors = []

    # Ensure data is a JSON string
    json_data = json.dumps(data)
    chunks = chunk_json_data_text(json_data)

    for i, chunk in enumerate(chunks):
        chunk = escape_curly_brackets(chunk)
        try:
            # embedding = model.embed_query(chunk)
            embeddings = pc.inference.embed(
                model="pinecone-sparse-english-v0",
                inputs=chunk,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            # print(f"Embeddings {embeddings}")
            sparse_values = embeddings.data[0]['sparse_values']
            sparse_indices = embeddings.data[0]['sparse_indices']

            doc_id = f"{model_name}_chunk_{i}_json_sparse"

            prompt = f"""
            Category: {model_name}
            {chunk}
            """

            expected_json = query_formatter(prompt, STRAPI_KEYWORD_GENERATOR_PROMPT_1)
            keywords = json.loads(expected_json).get("keywords", [])

            print(f"Generated keyword for chunk {i} model {model_name}: {keywords}")
            
            summary_response = query_formatter(chunk, STRAPI_SUMMARY_GENERATOR_PROMPT)

            vectors.append({
                "id": doc_id, 
                "sparse_values": {"indices": sparse_indices, "values": sparse_values}, 
                # "values": values,
                "metadata": {
                    "doc_id": doc_id, 
                    "doc_type": "text", 
                    "filename": doc_id, 
                    "summary": summary_response, 
                    "page_number": 1,
                    "keywords": keywords
                }
            })

            # Store the JSON data and raw chunk to a .txt file
            # with open(f"{doc_id}.txt", "w") as file:
            #     file.write(f"Raw Chunk:\n{chunk}\n\n")
            #     json.dump(vectors[-1], file, indent=4)

            # Upsert plaintext content to local DB
            upsert_docstore_in_db(doc_id, chunk)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    if vectors:
        index.upsert(vectors)


async def store_embeddings_in_pinecone_chunkjson_v2(index_name: str, data: Dict[str, str], model_name: str):
    print(f"Storing {model_name} embeddings in Pinecone index {index_name}")
    dimension = 1024
    create_index_if_not_exists(index_name, dimension=dimension)
    index = pc.Index(index_name)
    vectors = []

    for item in data:
        item_id = item.get("id", "")
        title = item.get("title", "")
        photos = item.get("photos", [])
        event_date = item.get("event_date", "")
        location = item.get("location", "")
        body = item.get("body", "")
        time = item.get("time", "")
        name = item.get("name", "")
        contact_number = item.get("contact_number", "")
        opening_hours = item.get("opening_hours", "")
        website = item.get("website", "")
        category = item.get("category", "")

        if model_name in ["event", "deal"]:
            content = f"Title: {title}\nImage URL: {photos[0] if photos else ''}\nDate: {event_date} {time}\nLocation: {location}\n{body}"
        elif model_name == "store":
            content = f"Store: {name}\nLocation: {location}\nContact: {contact_number}\nOpening Hours: {opening_hours}\nWebsite: {website}\nCategory: {category}\n{body}"
        else:
            content = ""
            for key, value in item.items():
                if key == "id":
                    continue
                content += f"{key.replace('_', ' ').title()}: {value}\n"

        chunks = chunk_json_data_text(content)

        for i, chunk in enumerate(chunks):
            chunk = escape_curly_brackets(chunk)
            try:
                embeddings = pc.inference.embed(
                    model="pinecone-sparse-english-v0",
                    inputs=chunk,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                sparse_values = embeddings.data[0]['sparse_values']
                sparse_indices = embeddings.data[0]['sparse_indices']

                doc_id = f"{model_name}_{item_id}_chunk_{i}_json"

                prompt = f"""
                Category: {model_name}
                {chunk}
                """

                expected_json = query_formatter(prompt, STRAPI_KEYWORD_GENERATOR_PROMPT_1)
                keywords = json.loads(expected_json).get("keywords", [])

                print(f"Generated keyword for chunk {i} model {model_name}: {keywords}")
                
                summary_response = query_formatter(chunk, STRAPI_SUMMARY_GENERATOR_PROMPT)

                metadata = {
                    "doc_id": doc_id, 
                    "doc_type": "text", 
                    "filename": doc_id, 
                    "summary": summary_response, 
                    "page_number": 1,
                    "keywords": keywords,
                    "category": model_name
                }

                if model_name in ["event", "deal"]:
                    metadata.update({
                        "event_date": event_date or "",
                        "location": location or ""
                    })
                elif model_name == "store":
                    metadata.update({
                        "type": category or "",
                        "location": location or ""
                    })

                vectors.append({
                    "id": doc_id, 
                    "sparse_values": {"indices": sparse_indices, "values": sparse_values}, 
                    "metadata": metadata
                })

                # Store the JSON data and raw chunk to a .txt file
                with open(f"{doc_id}.txt", "w") as file:
                    file.write(f"Raw Chunk:\n{chunk}\n\n")
                    json.dump(vectors[-1], file, indent=4)

                upsert_docstore_in_db(doc_id, chunk)
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")

    if vectors:
        print(f"Upserting {len(vectors)} vectors")
        index.upsert(vectors)
