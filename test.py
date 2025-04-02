import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load env variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION = "products"
VECTOR_INDEX_NAME = "vector_index"

print("🔧 MONGO_URI =", MONGO_URI)
print("📂 DB_NAME =", DB_NAME)

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


async def main():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME.lower()]

    # Check connection
    try:
        await db.command("ping")
        print("✅ Successfully connected to MongoDB Atlas")
    except Exception as e:
        print("❌ Could not connect to MongoDB:", e)
        return

    # List collections
    collections = await db.list_collection_names()
    print("📁 Collections in DB:", collections)

    # Sample one product
    sample_product = await db[COLLECTION].find_one({})
    print("🧪 Sample product document:", sample_product)

    # Prepare test query
    query_text = "organic milk"
    query_vector = model.encode(query_text).tolist()

    print("🔍 Running vector search with query:", query_text)

    # Run vector search safely
    try:
        results = await db[COLLECTION].aggregate([
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 10,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "product_num": 1,
                    "product_name": 1
                }
            }
        ]).to_list(length=5)

        print(f"✅ Vector search returned {len(results)} result(s).")
        for i, doc in enumerate(results):
            print(f"   {i+1}.", doc)

    except Exception as e:
        print("❌ Vector search failed!")
        print("🚨 Error:", str(e))


if __name__ == "__main__":
    asyncio.run(main())
