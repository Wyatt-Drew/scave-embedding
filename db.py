import motor.motor_asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Load env variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
if not MONGO_URI or not DB_NAME:
    raise ValueError("Missing MONGO_URI or DB_NAME environment variables.")

# Create MongoDB client and select database
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME.lower()]

def get_db():
    return db

async def connect_to_mongo():
    await db.command("ping")


# routes/products.py
from fastapi import APIRouter, HTTPException, Query
from db import get_db
from sentence_transformers import SentenceTransformer

router = APIRouter()
db = get_db()
model = SentenceTransformer("all-MiniLM-L6-v2")

@router.get("/ping")
async def ping():
    return {"status": "ok"}

@router.get("/products/SemanticSearch")
async def semantic_search(query: str = Query(...)):
    query_vector = model.encode(query).tolist()

    print("Incoming query:", query)
    print("Query vector (first 5 dims):", query_vector[:5])

    total_products = await db.products.count_documents({})
    with_embeddings = await db.products.count_documents({"embedding": {"$exists": True}})
    print(f"Total products: {total_products}, with embeddings: {with_embeddings}")

    try:
        similar_products = await db.products.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": [float(x) for x in query_vector],
                    "numCandidates": 300,
                    "limit": 10
                }
            }
        ]).to_list(length=10)
        print(f"Vector search returned {len(similar_products)} results.")
    except Exception as e:
        print("Vector search failed:", str(e))
        return {"error": str(e)}

    if not similar_products:
        print("No similar products found.")
        return []

    product_nums = [p["product_num"] for p in similar_products]
    print("Matching product_nums:", product_nums)

    latest_prices = await db.prices.aggregate([
        {"$match": {"product_num": {"$in": product_nums}}},
        {"$sort": {"date": -1}},
        {"$group": {
            "_id": {"product_num": "$product_num", "store_num": "$store_num"},
            "latest_price": {"$first": "$amount"},
            "latest_date": {"$first": "$date"},
            "unit": {"$first": "$unit"}
        }}
    ]).to_list(length=100)

    store_nums = list({p["_id"]["store_num"] for p in latest_prices})
    stores = await db.stores.find({"store_num": {"$in": store_nums}}).to_list(length=100)
    store_map = {s["store_num"]: s["store_name"] for s in stores}

    product_details = await db.products.find(
        {"product_num": {"$in": product_nums}},
        {"product_num": 1, "product_name": 1, "product_brand": 1, "product_link": 1, "image_url": 1, "description": 1}
    ).to_list(length=100)
    product_map = {p["product_num"]: p for p in product_details}

    response = []
    for price in latest_prices:
        pid = price["_id"]["product_num"]
        sid = price["_id"]["store_num"]
        product = product_map.get(pid, {})
        response.append({
            "product_num": pid,
            "store_num": sid,
            "store_name": store_map.get(sid, "Unknown Store"),
            "product_name": product.get("product_name", "Unknown Product"),
            "product_brand": product.get("product_brand", "Unknown Brand"),
            "product_link": product.get("product_link", ""),
            "image_url": product.get("image_url", ""),
            "description": product.get("description", ""),
            "latest_price": price["latest_price"],
            "latest_date": price["latest_date"],
            "unit": price["unit"]
        })

    print(f"Final response size: {len(response)}")
    return response
