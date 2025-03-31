import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
print("MONGO_URI =", MONGO_URI)
print("DB_NAME =", DB_NAME)

async def main():
    client = AsyncIOMotorClient(MONGO_URI)

    # Confirm connection
    print("✅ Connected to MongoDB")

    # List databases
    dbs = await client.list_database_names()
    print("Databases:", dbs)

    db = client[DB_NAME.lower()]

    # List collections
    collections = await db.list_collection_names()
    print("Collections in", DB_NAME, ":", collections)

    # Fetch one document from 'products'
    product = await db.products.find_one({})
    print("Product document:", product)

asyncio.run(main())
