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
from constants import STRAPI_KEYWORD_GENERATOR_PROMPT_1, STRAPI_SUMMARY_GENERATOR_PROMPT, STRAPI_SUMMARY_AND_DATE_GENERATOR_PROMPT, STRAPI_QUERY_DETECTOR_PROMPT, STRAPI_QUERY_DETECTOR_PROMPT_V2
from strapi_keyword_summary_generator import query_formatter
import datetime
from utils.parser import parse_event_dates

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
print(f"Pinecone Api key {api_key}")
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

    determined_query = determine_query(query, True)
    print(f"Determined query: {determined_query}")
    
    # Get date filters if it's an event/deal query
    keywords = determined_query.get("keywords", [])
    is_event_deal_query = determined_query.get("is_event_deal_query", False)
    start_date = determined_query.get("start_date", 0)
    end_date = determined_query.get("end_date", 0)
    
    # Prepare Pinecone filter if we have date criteria
    filter_dict = {
            "$and": [
                {"keywords": {"$in": keywords}}
            ]
        }
    if is_event_deal_query and start_date > 0 and end_date > 0:
        print(f"Filtering results by date range: {datetime.datetime.fromtimestamp(start_date).strftime('%Y-%m-%d')} to {datetime.datetime.fromtimestamp(end_date).strftime('%Y-%m-%d')}")
        filter_dict = {
            "$and": [
                {"keywords": {"$in": keywords}},
                {"category": {"$in": ["deal", "event"]}},
                {"start_date": {"$lte": end_date}},
                {"end_date": {"$gte": start_date}}
            ]
        }
        print(f"Using filter: {filter_dict}")

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
            include_values=False,
            filter=filter_dict
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

        event_dates = parse_event_dates(event_date)

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

                doc_id = f"{model_name}_{item_id}_chunk_{i}_sparse_v2"

                prompt = f"""
                Category: {model_name}
                {chunk}
                """

                expected_json = query_formatter(prompt, STRAPI_KEYWORD_GENERATOR_PROMPT_1)
                keywords = json.loads(expected_json).get("keywords", [])

                print(f"Generated keyword for chunk {i} model {model_name}: {keywords}")
                
                
                input = f"""
                Event Date: {event_date}
                Body: {prompt}
                """
                summary_date_response = query_formatter(input, STRAPI_SUMMARY_AND_DATE_GENERATOR_PROMPT)
                summary_date_response = json.loads(summary_date_response)
                print(f"Summary Date Response {summary_date_response}")
                
                # Extract values from response
                summary = summary_date_response.get("summary", "")
                response_start_date = summary_date_response.get("start_date", event_dates["start_date"])
                response_end_date = summary_date_response.get("end_date", event_dates["end_date"])
                
                # Validate the returned dates to ensure they're from the current year
                today = datetime.datetime.now()
                current_year = today.year
                
                # Create default timestamps
                today_timestamp = int(today.timestamp())
                end_of_year = datetime.datetime(current_year, 12, 31, 23, 59, 59)
                end_of_year_timestamp = int(end_of_year.timestamp())
                
                # Validate start date - check if it's from a past year
                if response_start_date > 0:
                    try:
                        start_date_dt = datetime.datetime.fromtimestamp(response_start_date)
                        if start_date_dt.year < current_year:
                            print(f"WARNING: Start date from past year detected: {response_start_date} ({start_date_dt.strftime('%Y-%m-%d')})")
                            # Use event_dates start_date as fallback, which should be current
                            response_start_date = event_dates["start_date"]
                            print(f"Corrected start_date to: {response_start_date} ({datetime.datetime.fromtimestamp(response_start_date).strftime('%Y-%m-%d')})")
                    except Exception as e:
                        print(f"Error validating start date: {e}, using default")
                        response_start_date = event_dates["start_date"]
                
                # Validate end date - check if it's from a past year
                if response_end_date > 0:
                    try:
                        end_date_dt = datetime.datetime.fromtimestamp(response_end_date)
                        if end_date_dt.year < current_year:
                            print(f"WARNING: End date from past year detected: {response_end_date} ({end_date_dt.strftime('%Y-%m-%d')})")
                            # Use event_dates end_date as fallback, which should be current
                            response_end_date = event_dates["end_date"]
                            print(f"Corrected end_date to: {response_end_date} ({datetime.datetime.fromtimestamp(response_end_date).strftime('%Y-%m-%d')})")
                    except Exception as e:
                        print(f"Error validating end date: {e}, using default")
                        response_end_date = event_dates["end_date"]
                
                # Use the validated dates
                start_date = response_start_date
                end_date = response_end_date

                metadata = {
                    "doc_id": doc_id, 
                    "doc_type": "text", 
                    "filename": doc_id, 
                    "summary": summary, 
                    "page_number": 1,
                    "keywords": keywords,
                    "category": model_name
                }

                if model_name in ["event", "deal"]:
                    metadata.update({
                        "event_date": event_date or "",
                        "start_date": start_date,
                        "end_date": end_date,
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
                with open(f"event_pinecone_data/{doc_id}.txt", "w") as file:
                    file.write(f"Raw Chunk:\n{chunk}\n\n")
                    json.dump(vectors[-1], file, indent=4)

                upsert_docstore_in_db(doc_id, chunk)
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")

    if vectors:
        print(f"Upserting {len(vectors)} vectors")
        index.upsert(vectors)



def determine_query(query: str, use_v2: bool = False):
    """
    Determine the intent of a user query and extract relevant date filters and keywords.
    
    Args:
        query: User query text
        use_v2: Whether to use the enhanced V2 prompt that also returns keywords
        
    Returns:
        Dictionary with query analysis including:
        - is_event_deal_query: Whether query is about events/deals
        - start_date: Unix timestamp for start date (or 0 if not applicable)
        - end_date: Unix timestamp for end date (or 0 if not applicable)
        - keywords: List of relevant keywords (only if use_v2=True)
    """
    start_time = datetime.datetime.now()

    # Choose which prompt to use
    prompt = STRAPI_QUERY_DETECTOR_PROMPT_V2 if use_v2 else STRAPI_QUERY_DETECTOR_PROMPT
    
    query_response = query_formatter(query, prompt)
    print(f"Query response {query_response}")
    
    try:
        query_response = json.loads(query_response)
        
        # Validate and correct timestamps
        # Get current date information
        today = datetime.datetime.now()
        current_year = today.year
        today_timestamp = int(today.timestamp())
        end_of_year = datetime.datetime(current_year, 12, 31, 23, 59, 59)
        end_of_year_timestamp = int(end_of_year.timestamp())
        
        # Check if the query is about events/deals
        is_event_deal_query = query_response.get("is_event_deal_query", False)
        
        # Get the existing timestamps
        start_date = query_response.get("start_date", 0)
        end_date = query_response.get("end_date", 0)
        
        # Validate timestamps for event/deal queries
        if is_event_deal_query:
            # If start date is suspicious (zero, negative, or from past year)
            if start_date <= 0 or datetime.datetime.fromtimestamp(start_date).year < current_year:
                print(f"WARNING: Invalid or past year start date detected: {start_date} ({datetime.datetime.fromtimestamp(start_date).strftime('%Y-%m-%d') if start_date > 0 else 'Invalid'})")
                # Replace with today's timestamp
                start_date = today_timestamp
                print(f"Corrected start_date to today: {start_date} ({datetime.datetime.fromtimestamp(start_date).strftime('%Y-%m-%d')})")
            
            # If end date is suspicious (zero, negative, or from past year)
            if end_date <= 0 or datetime.datetime.fromtimestamp(end_date).year < current_year:
                print(f"WARNING: Invalid or past year end date detected: {end_date} ({datetime.datetime.fromtimestamp(end_date).strftime('%Y-%m-%d') if end_date > 0 else 'Invalid'})")
                # Replace with end of current year timestamp
                end_date = end_of_year_timestamp
                print(f"Corrected end_date to end of year: {end_date} ({datetime.datetime.fromtimestamp(end_date).strftime('%Y-%m-%d')})")
            
            # Update the response with corrected timestamps
            query_response["start_date"] = start_date
            query_response["end_date"] = end_date
            
    except json.JSONDecodeError as e:
        print(f"Error parsing query response: {e}")
        # Return default response
        query_response = {
            "is_event_deal_query": False,
            "start_date": 0,
            "end_date": 0
        }
        if use_v2:
            query_response["keywords"] = []

    end_time = datetime.datetime.now()
    print(f"Time taken for determining query type: {end_time - start_time} seconds")
    return query_response