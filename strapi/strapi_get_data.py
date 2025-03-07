import requests
from constants import BASE_URL, PATHS
import json
from db.pinecone_db import store_embeddings_in_pinecone, store_embeddings_in_pinecone_chunkjson

def get_data_from_api(model_name):
    if model_name not in PATHS:
        raise ValueError(f"Model name '{model_name}' is not defined in PATHS.")
    
    url = f"{BASE_URL}{PATHS[model_name]}?populate=*"
    response = requests.get(url)
    
    if response.status_code != 200:
        response.raise_for_status()

    return response.json()

# Run the test
if __name__ == "__main__":
    # data = get_data_from_api("config-reward")
    # print(data)
    for model_name in PATHS.keys():
        datas = get_data_from_api(model_name)
        # print(f"Context retrieved for {model_name}: {json.dumps(datas, indent=2)}")
        print(f"Context retrieved for {model_name}")
        
        index_name = "i12katong-strapi-json"
        # store_embeddings_in_pinecone(index_name, datas, model_name)
        store_embeddings_in_pinecone_chunkjson(index_name, datas, model_name)