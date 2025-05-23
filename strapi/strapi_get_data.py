import requests
from constants import BASE_URL, PATHS, STAGING_INDEX_NAME, LOCAL_INDEX_NAME
import json
from db.pinecone_db import search_data_in_pinecone_sparse, store_embeddings_in_pinecone_chunkjson_v2
from db.pinecone_db_sparse_dense import search_data_in_pinecone_hybrid, store_embeddings_in_pinecone_hybrid, determine_query
import asyncio
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
import warnings
import datetime

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def strip_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def normalize_data(data):
    fields_to_keep = {
        'id', 'title', 'body', 'event_date', 'time', 'name', 'location', 'category', 'contact_number', 'website', 'opening_hours',
        'mall_address', 'phone_customer_service', 'counter_operating_hours', 'summary', 'getting_by_bus', 'getting_by_car',
        'getting_by_train', 'concierge_services', 'amentities_wifi', 'amentities_nursing_room', 'amentities_charging_point',
        'carpark_charges_car', 'carpark_charges_motorcycle', 'about', 'faq', 'terms', 'terms_of_use', 'data_protection_policy',
        'general_privacy_notice', 'whistleblowing_policy', 'photos'
    }
    normalized_data = []

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("Expected data to be a list of dictionaries or a single dictionary")

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Expected each item in data to be a dictionary")
        normalized_item = {key: strip_html_tags(value) if isinstance(value, str) else value 
                           for key, value in item.items() if key in fields_to_keep and value is not None}
        
        # Extract photos.url if photos exist
        if 'photos' in item and isinstance(item['photos'], list):
            photos_urls = [photo['url'] for photo in item['photos'] if 'url' in photo]
            normalized_item = {k: v for k, v in sorted(normalized_item.items(), key=lambda x: (x[0] not in ['id', 'title', 'body'], x[0]))}
            normalized_item = {k: v for k, v in normalized_item.items() if k != 'photos'}
            normalized_item = {**{k: normalized_item[k] for k in ['id', 'title'] if k in normalized_item}, 'photos': photos_urls, **{k: normalized_item[k] for k in normalized_item if k not in ['id', 'title']}}

        normalized_data.append(normalized_item)

    return normalized_data

def get_data_from_api(model_name):
    if model_name not in PATHS:
        raise ValueError(f"Model name '{model_name}' is not defined in PATHS.")
    
    url = f"{BASE_URL}{PATHS[model_name]}?populate=*&pagination[pageSize]=1000&status=published"
    response = requests.get(url)
    
    if response.status_code != 200:
        response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict) or 'data' not in data:
        raise ValueError("Unexpected API response format")
    return normalize_data(data['data'])

async def main():
    for model_name in PATHS.keys():
        datas = get_data_from_api(model_name)
        # print(f"Context retrieved for {model_name}")
        # print(f"Data {datas}")
        
        # Store the datas to .txt for each model_name
        # with open(f"{model_name}_data.txt", "w") as file:
        #     for data in datas:
        #         file.write(json.dumps(data) + "\n")
        
        index_name = STAGING_INDEX_NAME
        index_name = LOCAL_INDEX_NAME
        index_name = "i12katong-strapi-sparse-dense"
        # await store_embeddings_in_pinecone_hybrid(index_name, datas, model_name)

        # query = "show me promo for this week only"
        # query = "where is ippudo located?"
        # query="any promo for this week?"
        # query = "current event"
        # resp = search_data_in_pinecone_sparse(index_name=index_name, query=query, k=20)
        # print(f"Retrieved docs {resp}")


        # Example usage
        # query = "show me promo for this week only"
        # query = "where is ippudo located?"
        query = "tell me about coucou promo"
        # query = "tell me about activities"
        start_date = datetime.datetime.now()
        resp = determine_query(query, use_v2=True)
        print(f"Determined query {resp}")
        end_date = datetime.datetime.now()
        print(f"Time taken for search: {end_date - start_date}")
        # this_week_promos = search_data_in_pinecone_hybrid(index_name, query, use_enhanced_query=False)
        # print(f"Retrieved docs {this_week_promos}")

        # # Display results
        # for i, promo in enumerate(this_week_promos, 1):
        #     print(f"\n--- Promo {i} ---")
        #     print(f"Score: {promo['score']}")
        #     print(f"ID: {promo['id']}")
        #     print(f"Summary: {promo['summary']}")
        #     print(f"Location: {promo['location']}")
        #     print(f"Event Date: {promo['event_date']}")
        #     print(f"Keywords: {', '.join(promo['keywords'])}")

        # With custom alpha weighting (more weight to sparse/keyword matching)
        # results = search_promos_by_date_range(
        #     index_name=index_name, 
        #     query="any promo for june?",
        #     # date_range="2025-03-17 to 2025-03-23",
        #     date_range=None,
        #     k=50,
        #     alpha=0.7  # More emphasis on keyword matching
        # )

        # # Display results
        # for i, promo in enumerate(results, 1):
        #     print(f"\n--- Result {i} ---")
        #     print(f"ID: {promo['id']}")
        #     print(f"Score: {promo['score']}")
        #     print(f"Summary: {promo['summary']}")
        #     print(f"Location: {promo['location']}")
        #     print(f"Event Date: {promo['event_date']}")

if __name__ == "__main__":
    asyncio.run(main())