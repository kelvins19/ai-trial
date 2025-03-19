# Import the Pinecone library
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Union, Optional
from dotenv import load_dotenv
import os
import json
import datetime
import re
from langchain.text_splitter import (
    MarkdownTextSplitter, 
    RecursiveJsonSplitter, 
    RecursiveCharacterTextSplitter
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from .db import upsert_docstore_in_db
from constants import (
    STRAPI_KEYWORD_GENERATOR_PROMPT_1, 
    STRAPI_SUMMARY_GENERATOR_PROMPT, 
    STRAPI_SUMMARY_AND_DATE_GENERATOR_PROMPT,
    STRAPI_QUERY_DETECTOR_PROMPT,
    STRAPI_QUERY_DETECTOR_PROMPT_V2
)
from strapi_keyword_summary_generator import query_formatter
from utils.parser import parse_event_dates

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
dense_model_name = os.getenv("DENSE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embed_access_key = "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"

# Initialize Pinecone and embedding models
pc = Pinecone(api_key=api_key)
model = HuggingFaceInferenceAPIEmbeddings(model_name=model_name, api_key=embed_access_key)
dense_model = SentenceTransformer(dense_model_name)

#------------------------------------------------------------------------------
# Index Management Functions
#------------------------------------------------------------------------------

def create_index_if_not_exists(index_name: str, dimension: int = 384):
    """
    Create a hybrid index for both sparse and dense vectors if it doesn't exist.
    
    Args:
        index_name: Name of the Pinecone index to create
        dimension: Dimensionality of dense vectors (default: 384)
    """
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        # Create a hybrid index for both sparse and dense vectors
        # For hybrid search, we need to use pod_type="s1" or "p1" and metric="dotproduct"
        pc.create_index(
            name=index_name,
            dimension=dimension,  # dimensionality of dense vectors
            metric="dotproduct",  # Required for hybrid search
            # Either use pod_type for dedicated deployments
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new hybrid index: {index_name} with dimension {dimension}")

#------------------------------------------------------------------------------
# Data Processing Utilities
#------------------------------------------------------------------------------

def escape_curly_brackets(text: str) -> str:
    """Escape curly brackets in text to prevent formatting issues."""
    return text.replace("{", "{{").replace("}", "}}")

def format_json_to_markdown(json_data) -> str:
    """
    Convert JSON data to a markdown formatted string.
    
    Args:
        json_data: JSON data as dictionary or list
        
    Returns:
        Markdown formatted string representation
    """
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
    """
    Split JSON data into chunks that fit within metadata size limits.
    
    Args:
        json_data: JSON data to chunk
        max_metadata_size: Maximum size in bytes for each chunk
        
    Returns:
        List of chunked JSON data
    """
    splitter = RecursiveJsonSplitter(min_chunk_size=10, max_chunk_size=max_metadata_size)
    chunks = splitter.split_text(json_data, True, True)
    return chunks

def chunk_json_data_text(json_data):
    """
    Split JSON data into text chunks with defined size and overlap.
    
    Args:
        json_data: JSON data to chunk (dict or string)
        
    Returns:
        List of text chunks
    """
    if isinstance(json_data, dict):
        json_data = json.dumps(json_data)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_text(json_data)
    return chunks

#------------------------------------------------------------------------------
# Embedding Generation Functions
#------------------------------------------------------------------------------

def get_sparse_embeddings(text: str):
    """
    Generate sparse embeddings for a given text using Pinecone's sparse encoder.
    This uses a BM25-based encoder for efficient keyword matching.
    
    Args:
        text: Input text to encode
        
    Returns:
        Dictionary with indices and values for sparse representation
    """
    embeddings = pc.inference.embed(
        model="pinecone-sparse-english-v0",  # Pinecone's sparse encoder - BM25 style
        inputs=text,
        parameters={
            "input_type": "passage" if len(text) > 100 else "query", 
            "truncate": "END"
        }
    )
    return {
        "indices": embeddings.data[0]['sparse_indices'],
        "values": embeddings.data[0]['sparse_values']
    }

def get_dense_embeddings(text: str) -> List[float]:
    """
    Generate dense embeddings for a given text using SentenceTransformer.
    
    Args:
        text: Input text to encode
        
    Returns:
        List of float values representing the dense embedding
    """
    embedding = dense_model.encode(text)
    return embedding.tolist()

#------------------------------------------------------------------------------
# Search Functions
#------------------------------------------------------------------------------

def search_data_in_pinecone_hybrid(
    index_name: str, 
    query: str, 
    k: int = 20,
    alpha: float = 0.5,  # Weight between sparse (0) and dense (1)
    use_enhanced_query: bool = True  # Whether to use the V2 prompt with keywords
):
    """
    Search data using both sparse and dense vectors with controllable weighting.
    
    Args:
        index_name: Name of the Pinecone index
        query: Search query text
        k: Number of results to return
        alpha: Weight between sparse (0) and dense (1)
            - alpha=0: completely sparse (exact term matching)
            - alpha=1: completely dense (semantic search)
            - alpha=0.5: balanced hybrid search
        use_enhanced_query: Whether to use the enhanced V2 query detection with keywords
            
    Returns:
        Search results from Pinecone
    """
    start_time = datetime.datetime.now()
    
    # Determine query intent and extract date filters (and keywords if V2)
    determined_query = determine_query(query, use_v2=use_enhanced_query)
    print(f"Determined query: {determined_query}")
    
    # Get date filters if it's an event/deal query
    is_event_deal_query = determined_query.get("is_event_deal_query", False)
    start_date = determined_query.get("start_date", 0)
    end_date = determined_query.get("end_date", 0)
    
    # Get keywords if using V2 prompt
    keywords = determined_query.get("keywords", [])
    
    # If we have keywords from the V2 prompt, enhance the query with them
    enhanced_query = query
    if use_enhanced_query and keywords:
        # Add the top 3 most relevant keywords to the query to improve search
        # This works because sparse vectors (BM25) are excellent at term matching
        enhanced_keywords = " ".join(keywords[:3])
        enhanced_query = f"{query} {enhanced_keywords}"
        print(f"Enhanced query with keywords: '{enhanced_query}'")
    
    # Get sparse embeddings using BM25 sparse encoder
    sparse_embed = get_sparse_embeddings(enhanced_query)
    
    # Get dense embeddings - use original query for semantic search
    # as adding keywords might skew the semantic meaning
    dense_embed = get_dense_embeddings(query)
    
    end_time = datetime.datetime.now()
    print(f"Time taken for vectorization: {end_time - start_time} seconds")
    
    # Prepare Pinecone filter if we have date criteria
    filter_dict = None
    if is_event_deal_query and start_date > 0 and end_date > 0:
        print(f"Filtering results by date range: {datetime.datetime.fromtimestamp(start_date).strftime('%Y-%m-%d')} to {datetime.datetime.fromtimestamp(end_date).strftime('%Y-%m-%d')}")
        filter_dict = {
            "$and": [
                {"category": {"$in": ["deal", "event"]}},
                {"start_date": {"$lte": str(end_date)}},
                {"end_date": {"$gte": str(start_date)}}
            ]
        }
        print(f"Using filter: {filter_dict}")

    start_time = datetime.datetime.now()
    index = pc.Index(index_name)
    
    # Hybrid search using both sparse and dense vectors
    # The alpha parameter controls the weight between sparse and dense:
    # alpha=0 means pure sparse (BM25), alpha=1 means pure dense (semantic)
    retrieved_docs = index.query(
        namespace="",
        vector=dense_embed,
        sparse_vector={
            "indices": sparse_embed["indices"],
            "values": sparse_embed["values"]
        },
        top_k=k,
        include_metadata=True,
        include_values=False,
        alpha=alpha,  # Weighting parameter
        filter=filter_dict  # Apply filter directly in query
    )
    
    end_time = datetime.datetime.now()
    print(f"Time taken for hybrid retrieval from pinecone: {end_time - start_time} seconds")
    
    # If we've filtered at query time, we don't need to post-process
    # But keeping the code for backward compatibility
    if is_event_deal_query and start_date > 0 and end_date > 0 and filter_dict is None:
        filtered_results = []
        for match in retrieved_docs.matches:
            metadata = match.metadata
            # Check if it's a deal/promo or event
            if metadata.get("category") in ["deal", "event"]:
                # Get date information
                item_start_date = metadata.get("start_date")
                item_end_date = metadata.get("end_date")
                
                # Check if the item is active during the requested date range
                if item_start_date and item_end_date:
                    # Item must start before end of requested period and end after start of requested period
                    if int(item_start_date) <= end_date and int(item_end_date) >= start_date:
                        filtered_results.append(match)
        
        # Create a new response with filtered matches
        if hasattr(retrieved_docs, "_replace"):
            # If retrieved_docs is a namedtuple (which is common in Pinecone's response)
            filtered_docs = retrieved_docs._replace(matches=filtered_results)
            print(f"Filtered from {len(retrieved_docs.matches)} to {len(filtered_results)} results")
            return filtered_docs
        else:
            # Alternative fallback if the response structure is different
            retrieved_docs.matches = filtered_results
            print(f"Filtered from {len(retrieved_docs.matches)} to {len(filtered_results)} results")
            return retrieved_docs
            
    return retrieved_docs

def search_data_in_pinecone_sparse(index_name: str, query: str, k: int = 20):
    """
    Legacy function for sparse-only search.
    
    Args:
        index_name: Name of the Pinecone index
        query: Search query text
        k: Number of results to return
        
    Returns:
        Search results from Pinecone using only sparse vectors
    """
    start_time = datetime.datetime.now()
    embeddings = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=query,
        parameters={"input_type": "query"}
    )
    sparse_values = embeddings.data[0]['sparse_values']
    sparse_indices = embeddings.data[0]['sparse_indices']
    end_time = datetime.datetime.now()
    print(f"Time taken for convert to vector: {end_time - start_time} seconds")

    # Determine query intent and extract date filters
    determined_query = determine_query(query)
    print(f"Determined query: {determined_query}")
    
    # Get date filters if it's an event/deal query
    is_event_deal_query = determined_query.get("is_event_deal_query", False)
    start_date = determined_query.get("start_date", 0)
    end_date = determined_query.get("end_date", 0)
    
    # Prepare Pinecone filter if we have date criteria
    filter_dict = None
    if is_event_deal_query and start_date > 0 and end_date > 0:
        print(f"Filtering results by date range: {datetime.datetime.fromtimestamp(start_date).strftime('%Y-%m-%d')} to {datetime.datetime.fromtimestamp(end_date).strftime('%Y-%m-%d')}")
        filter_dict = {
            "$and": [
                {"category": {"$in": ["deal", "event"]}},
                {"start_date": {"$lte": str(end_date)}},
                {"end_date": {"$gte": str(start_date)}}
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

def search_data_in_pinecone_dense(index_name: str, query: str, k: int = 20):
    """
    Dense-only search function.
    
    Args:
        index_name: Name of the Pinecone index
        query: Search query text
        k: Number of results to return
        
    Returns:
        Search results from Pinecone using only dense vectors
    """
    start_time = datetime.datetime.now()
    dense_embedding = get_dense_embeddings(query)
    end_time = datetime.datetime.now()
    print(f"Time taken for dense vectorization: {end_time - start_time} seconds")

    start_time = datetime.datetime.now()
    index = pc.Index(index_name)
    retrieved_docs = index.query(
        namespace="",
        vector=dense_embedding,
        top_k=k,
        include_metadata=True,
        include_values=False
    )
    
    end_time = datetime.datetime.now()
    print(f"Time taken for dense retrieval from pinecone: {end_time - start_time} seconds")
    return retrieved_docs

def search_data_in_pinecone(index_name: str, query: str, k: int = 20): 
    """
    Default search function that uses hybrid search.
    
    Args:
        index_name: Name of the Pinecone index
        query: Search query text
        k: Number of results to return
        
    Returns:
        Search results from Pinecone using hybrid search
    """
    # Use hybrid search as the default search method
    return search_data_in_pinecone_hybrid(index_name, query, k)

#------------------------------------------------------------------------------
# Specialized Search Functions
#------------------------------------------------------------------------------

def search_promos_for_this_week(index_name: str, query: str, k: int = 50, alpha: float = 0.5):
    """
    Search for promos/deals that are active during the current week using hybrid search.
    
    Args:
        index_name: Name of the Pinecone index
        query: User query (e.g., "show me promo for this week only")
        k: Maximum number of results to retrieve initially
        alpha: Weight between sparse (0) and dense (1)
    
    Returns:
        List of deals that match both the query and are active this week
    """
    # First, clean the query to focus on the promo content
    # Remove the time-specific parts like "for this week only"
    search_query = query.replace("for this week only", "").replace("show me", "").strip()
    if not search_query:
        search_query = "promotion deal discount"  # Default search if query is too generic
        
    # Get current date information
    today = datetime.datetime.now()
    # Calculate start of week (Monday)
    start_of_week = today - datetime.timedelta(days=today.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    # Calculate end of week (Sunday)
    end_of_week = start_of_week + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    # Convert to Unix timestamps
    start_timestamp = int(start_of_week.timestamp())
    end_timestamp = int(end_of_week.timestamp())
    
    print(f"Searching for: '{search_query}'")
    print(f"Date range: {start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}")
    
    # Search Pinecone using hybrid search
    search_results = search_data_in_pinecone_hybrid(index_name, search_query, k=k, alpha=alpha)
    
    # Filter results by date
    filtered_results = []
    for match in search_results.matches:
        metadata = match.metadata
        # Check if it's a deal/promo
        if metadata.get("category") == "deal":
            # Get date information
            start_date = metadata.get("start_date")
            end_date = metadata.get("end_date")
            
            # Check if the promo is active during this week
            if start_date and end_date:
                # Promo must start before end of week and end after start of week
                if int(start_date) <= end_timestamp and int(end_date) >= start_timestamp:
                    filtered_results.append({
                        "score": match.score,
                        "id": match.id,
                        "summary": metadata.get("summary", ""),
                        "location": metadata.get("location", ""),
                        "event_date": metadata.get("event_date", ""),
                        "keywords": metadata.get("keywords", []),
                        "chunk_text": metadata.get("chunk_text", "")
                    })
    
    print(f"Found {len(filtered_results)} promos active this week")
    return filtered_results

def search_promos_by_date_range(
    index_name: str, 
    query: str, 
    date_range: str = "this week",
    location: str = None,
    k: int = 50, 
    alpha: float = 0.5,
    use_enhanced_query: bool = True  # Whether to use the V2 prompt with keywords
):
    """
    Search for promos/deals that are active during a specific date range using hybrid search.
    With the enhanced query option, it will automatically extract keywords to improve search accuracy.
    
    Args:
        index_name: Name of the Pinecone index
        query: User query (e.g., "show me hotel promotions")
        date_range: Text description of date range like "this week", "next week", "weekend", 
                   "this month", or specific dates like "2023-12-25 to 2023-12-31"
                   Set to None to disable date filtering.
        location: Optional location filter
        k: Maximum number of results to retrieve initially
        alpha: Weight between sparse (0) and dense (1)
        use_enhanced_query: Whether to use the enhanced V2 prompt with keywords
    
    Returns:
        List of deals that match both the query and date criteria
    """
    # First, clean the query to focus on the promo content
    search_query = query.replace("for this week only", "").replace("show me", "").strip()
    
    # Determine query intent and extract date filters (and keywords if V2)
    determined_query = determine_query(search_query, use_v2=use_enhanced_query)
    
    # Get keywords if using V2 prompt
    keywords = determined_query.get("keywords", [])
    
    # Enhance the query with keywords if available
    enhanced_query = search_query
    if use_enhanced_query and keywords:
        # Use the top keywords to enhance the query
        enhanced_keywords = " ".join(keywords[:5])  # Use top 5 keywords
        enhanced_query = f"{search_query} {enhanced_keywords}"
        print(f"Enhanced query with keywords: '{enhanced_query}'")
    
    # Add location to query if provided
    if location and location.lower() not in enhanced_query.lower():
        enhanced_query = f"{enhanced_query} {location}"
        
    if not enhanced_query:
        enhanced_query = "promotion deal discount"  # Default search if query is too generic
    
    # Parse date range if provided
    start_timestamp = None
    end_timestamp = None
    
    if date_range is not None:
        today = datetime.datetime.now()
        
        # Default to this week
        start_date = today - datetime.timedelta(days=today.weekday())
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
        
        # Handle specific date range patterns
        if date_range == "this week":
            # Already set to default
            pass
        elif date_range == "next week":
            start_date = start_date + datetime.timedelta(days=7)
            end_date = end_date + datetime.timedelta(days=7)
        elif date_range == "weekend":
            # Set to upcoming weekend (Friday to Sunday)
            days_until_friday = (4 - today.weekday()) % 7
            start_date = today + datetime.timedelta(days=days_until_friday)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + datetime.timedelta(days=2, hours=23, minutes=59, seconds=59)
        elif date_range == "this month":
            # Set to current month
            start_date = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Last day of month calculation
            if today.month == 12:
                end_date = today.replace(year=today.year+1, month=1, day=1) - datetime.timedelta(days=1)
            else:
                end_date = today.replace(month=today.month+1, day=1) - datetime.timedelta(days=1)
            end_date = end_date.replace(hour=23, minute=59, second=59)
        elif date_range == "next month":
            # Set to next month
            if today.month == 12:
                start_date = today.replace(year=today.year+1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date.replace(month=2, day=1) - datetime.timedelta(days=1)
            else:
                start_date = today.replace(month=today.month+1, day=1, hour=0, minute=0, second=0, microsecond=0)
                if today.month == 11:  # December
                    end_date = today.replace(year=today.year+1, month=1, day=1) - datetime.timedelta(days=1)
                else:
                    end_date = today.replace(month=today.month+2, day=1) - datetime.timedelta(days=1)
            end_date = end_date.replace(hour=23, minute=59, second=59)
        else:
            # Try to parse "YYYY-MM-DD to YYYY-MM-DD" format
            date_pattern = r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})"
            match = re.search(date_pattern, date_range)
            if match:
                try:
                    start_str, end_str = match.groups()
                    start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d")
                    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d")
                    end_date = end_date.replace(hour=23, minute=59, second=59)
                except (ValueError, TypeError):
                    # If parsing fails, keep default values
                    pass
        
        # Convert to Unix timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        print(f"Searching for: '{enhanced_query}'")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    else:
        print(f"Searching for: '{enhanced_query}' (no date filtering)")
    
    # Search Pinecone using hybrid search
    search_results = search_data_in_pinecone_hybrid(
        index_name, 
        enhanced_query, 
        k=k, 
        alpha=alpha, 
        use_enhanced_query=False  # Don't use enhanced query again as we've already done it
    )
    
    # Filter results
    filtered_results = []
    for match in search_results.matches:
        metadata = match.metadata
        # Check if it's a deal/promo
        if metadata.get("category") == "deal":
            # Apply date filter if enabled
            date_match = True
            if start_timestamp is not None and end_timestamp is not None:
                # Get date information
                start_date_ts = metadata.get("start_date")
                end_date_ts = metadata.get("end_date")
                
                # Check if the promo is active during selected date range
                if start_date_ts and end_date_ts:
                    date_match = int(start_date_ts) <= end_timestamp and int(end_date_ts) >= start_timestamp
            
            # Apply location filter if provided
            location_match = True
            if location:
                promo_location = metadata.get("location", "").lower()
                location_match = location.lower() in promo_location
            
            if date_match and location_match:
                filtered_results.append({
                    "score": match.score,
                    "id": match.id,
                    "summary": metadata.get("summary", ""),
                    "location": metadata.get("location", ""),
                    "event_date": metadata.get("event_date", ""),
                    "keywords": metadata.get("keywords", []),
                    "chunk_text": metadata.get("chunk_text", "")
                })
    
    if date_range is not None:
        print(f"Found {len(filtered_results)} matching promos for the date range")
    else:
        print(f"Found {len(filtered_results)} matching promos")
    
    return filtered_results

#------------------------------------------------------------------------------
# Vector Storage Functions
#------------------------------------------------------------------------------

async def store_embeddings_in_pinecone_hybrid(index_name: str, data: Dict[str, str], model_name: str):
    """
    Store both sparse and dense embeddings for the given data.
    
    Args:
        index_name: Name of the Pinecone index
        data: Data to embed and store
        model_name: Category/type of the data being stored
        
    Returns:
        None
    """
    print(f"Storing hybrid embeddings for {model_name} in Pinecone index {index_name}")
    dimension = 384  # Dimension of the dense model
    create_index_if_not_exists(index_name, dimension=dimension)
    index = pc.Index(index_name)
    vectors = []

    # Ensure data is a JSON string
    json_data = json.dumps(data)
    chunks = chunk_json_data_text(json_data)

    for i, chunk in enumerate(chunks):
        chunk = escape_curly_brackets(chunk)
        try:
            # Generate sparse embeddings
            sparse_embed = get_sparse_embeddings(chunk)
            
            # Generate dense embeddings
            dense_embed = get_dense_embeddings(chunk)

            doc_id = f"{model_name}_chunk_{i}_hybrid"

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
                "values": dense_embed,  # Dense vector
                "sparse_values": {  # Sparse vector
                    "indices": sparse_embed["indices"], 
                    "values": sparse_embed["values"]
                }, 
                "metadata": {
                    "doc_id": doc_id, 
                    "doc_type": "text", 
                    "filename": doc_id, 
                    "summary": summary_response, 
                    "page_number": 1,
                    "keywords": keywords,
                    "chunk_text": chunk,
                    "category": model_name
                }
            })

            # Upsert plaintext content to local DB
            upsert_docstore_in_db(doc_id, chunk)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    if vectors:
        print(f"Upserting {len(vectors)} hybrid vectors")
        index.upsert(vectors)

async def store_embeddings_in_pinecone_chunkjson_v2_hybrid(index_name: str, data: Dict[str, str], model_name: str):
    """
    Store both sparse and dense embeddings for structured data like events and deals.
    
    Args:
        index_name: Name of the Pinecone index
        data: Structured data (list of items) to embed and store
        model_name: Category/type of the data ("event", "deal", "store", etc.)
        
    Returns:
        None
    """
    print(f"Storing hybrid embeddings for {model_name} in Pinecone index {index_name}")
    dimension = 384  # Dimension of the dense model
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

        # Format content based on model type
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
                # Generate sparse embeddings
                sparse_embed = get_sparse_embeddings(chunk)
                
                # Generate dense embeddings
                dense_embed = get_dense_embeddings(chunk)

                doc_id = f"{model_name}_{item_id}_chunk_{i}_hybrid"

                prompt = f"""
                Category: {model_name}
                {chunk}
                """

                # Generate keywords
                expected_json = query_formatter(prompt, STRAPI_KEYWORD_GENERATOR_PROMPT_1)
                keywords = json.loads(expected_json).get("keywords", [])
                print(f"Generated keyword for chunk {i} model {model_name}: {keywords}")
                
                # Generate summary with dates
                input_text = f"""
                Event Date: {event_date}
                Body: {prompt}
                """
                summary_date_response = query_formatter(input_text, STRAPI_SUMMARY_AND_DATE_GENERATOR_PROMPT)
                summary_date_response = json.loads(summary_date_response)
                summary = summary_date_response.get("summary", "")
                start_date = summary_date_response.get("start_date", event_dates["start_date"])
                end_date = summary_date_response.get("end_date", event_dates["end_date"])

                # Create metadata
                metadata = {
                    "doc_id": doc_id, 
                    "doc_type": "text", 
                    "filename": doc_id, 
                    "summary": summary, 
                    "page_number": 1,
                    "keywords": keywords,
                    "category": model_name,
                    "chunk_text": chunk
                }

                # Add type-specific metadata
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

                # Create vector with both sparse and dense embeddings
                vectors.append({
                    "id": doc_id, 
                    "values": dense_embed,  # Dense vector
                    "sparse_values": {  # Sparse vector
                        "indices": sparse_embed["indices"], 
                        "values": sparse_embed["values"]
                    }, 
                    "metadata": metadata
                })

                # Store debugging data
                output_dir = "event_pinecone_data"
                os.makedirs(output_dir, exist_ok=True)
                with open(f"{output_dir}/{doc_id}.txt", "w") as file:
                    file.write(f"Raw Chunk:\n{chunk}\n\n")
                    json.dump(vectors[-1], file, indent=4)

                # Store in document DB
                upsert_docstore_in_db(doc_id, chunk)
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")

    # Upsert vectors in batches
    if vectors:
        print(f"Upserting {len(vectors)} hybrid vectors")
        index.upsert(vectors)

#------------------------------------------------------------------------------
# Backward Compatibility Functions
#------------------------------------------------------------------------------

async def store_embeddings_in_pinecone_chunkjson(index_name: str, data: Dict[str, str], model_name: str):
    """Backward compatibility wrapper for store_embeddings_in_pinecone_hybrid."""
    return await store_embeddings_in_pinecone_hybrid(index_name, data, model_name)

async def store_embeddings_in_pinecone_chunkjson_v2(index_name: str, data: Dict[str, str], model_name: str):
    """Backward compatibility wrapper for store_embeddings_in_pinecone_chunkjson_v2_hybrid."""
    return await store_embeddings_in_pinecone_chunkjson_v2_hybrid(index_name, data, model_name)

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
    query_response = json.loads(query_response)
    
    end_time = datetime.datetime.now()
    print(f"Time taken for determining query type: {end_time - start_time} seconds")
    return query_response

#------------------------------------------------------------------------------
# Example Usage Functions
#------------------------------------------------------------------------------

def search_with_enhanced_query(index_name: str, user_query: str, k: int = 20):
    """
    Example function demonstrating the enhanced query detection capabilities.
    This function shows how to use the V2 query detector to improve search results.
    
    Args:
        index_name: Name of the Pinecone index
        user_query: The raw query from the user
        k: Number of results to return
        
    Returns:
        A dictionary with search results and analysis
    """
    print(f"\n===== Enhanced Query Search =====")
    print(f"Original query: '{user_query}'")
    
    # First, get the enhanced query analysis (date range + keywords)
    analysis = determine_query(user_query, use_v2=True)
    
    is_event_deal_query = analysis.get("is_event_deal_query", False)
    keywords = analysis.get("keywords", [])
    start_date = analysis.get("start_date", 0)
    end_date = analysis.get("end_date", 0)
    
    print(f"\nQuery Analysis:")
    print(f"- Is event/deal query: {is_event_deal_query}")
    if keywords:
        print(f"- Generated keywords: {', '.join(keywords[:5])}...")
    if is_event_deal_query and start_date > 0 and end_date > 0:
        start_date_str = datetime.datetime.fromtimestamp(start_date).strftime('%Y-%m-%d')
        end_date_str = datetime.datetime.fromtimestamp(end_date).strftime('%Y-%m-%d')
        print(f"- Date range: {start_date_str} to {end_date_str}")
    
    # Create an enhanced query using the top keywords
    enhanced_query = user_query
    if keywords:
        # Use the top keywords to enhance the search query
        enhanced_keywords = " ".join(keywords[:3])  # Use top 3 keywords
        enhanced_query = f"{user_query} {enhanced_keywords}"
    
    print(f"\nEnhanced search query: '{enhanced_query}'")
    
    # Perform the search with the enhanced query
    search_results = search_data_in_pinecone_hybrid(
        index_name=index_name,
        query=enhanced_query,
        k=k,
        alpha=0.5,  # Balanced hybrid search
        use_enhanced_query=False  # We've already enhanced the query
    )
    
    # Format the results into a more readable format
    formatted_results = []
    for i, match in enumerate(search_results.matches[:min(5, len(search_results.matches))]):
        metadata = match.metadata
        formatted_results.append({
            "position": i+1,
            "id": match.id,
            "score": round(match.score, 3),
            "summary": metadata.get("summary", "")[:100] + "..." if metadata.get("summary") else "",
            "category": metadata.get("category", ""),
            "location": metadata.get("location", ""),
        })
    
    print(f"\nTop results:")
    for result in formatted_results:
        print(f"{result['position']}. [{result['category']}] {result['summary']} (score: {result['score']})")
    
    print(f"\nFound {len(search_results.matches)} total results")
    
    return {
        "analysis": analysis,
        "enhanced_query": enhanced_query,
        "results_count": len(search_results.matches),
        "top_results": formatted_results
    }