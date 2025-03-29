from fastapi import FastAPI, Query
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from typing import List
import uvicorn

app = FastAPI()

# Load model and DB
model = SentenceTransformer("all-MiniLM-L6-v2")
client = MongoClient("your_mongo_uri")
collection = client["test"]["products"]

@app.get("/search/")
def semantic_search(query: str = Query(..., min_length=1), limit: int = 10):
    query_vector = model.encode(query).tolist()
    
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 300,
                "limit": limit
            }
        }
    ])
    
    output = []
    for doc in results:
        output.append({
            "product_num": doc.get("product_num"),
            "product_name": doc.get("product_name"),
            "product_brand": doc.get("product_brand"),
            "description": doc.get("description"),
            "product_link": doc.get("product_link"),
            "image_url": doc.get("image_url")
        })

    return output

# Uncomment this line to run directly
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
