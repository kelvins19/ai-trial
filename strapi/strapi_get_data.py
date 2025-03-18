import requests
from constants import BASE_URL, PATHS, STAGING_INDEX_NAME, LOCAL_INDEX_NAME
import json
from db.pinecone_db import search_data_in_pinecone, store_embeddings_in_pinecone_chunkjson, search_data_in_pinecone_sparse, store_embeddings_in_pinecone_chunkjson_v2
import asyncio
from bs4 import BeautifulSoup

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
    
    url = f"{BASE_URL}{PATHS[model_name]}?populate=*"
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
        index_name = "i12katong-strapi-sparse-v2"
        await store_embeddings_in_pinecone_chunkjson_v2(index_name, datas, model_name)

        # resp = search_data_in_pinecone_sparse(index_name=index_name, query="participating stores in weekday dine and delight", k=20)
        # print(f"Retrieved docs {resp}")

if __name__ == "__main__":
    asyncio.run(main())