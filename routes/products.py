from fastapi import APIRouter, Query
from db import get_db
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
from operator import itemgetter
import re

router = APIRouter()
db = get_db()
model = SentenceTransformer("all-MiniLM-L6-v2")


@router.get("/products/SemanticSearch")
async def semantic_search(query: str = Query(...)):
    query_vector = model.encode(query).tolist()

    try:
        similar_products = await db.products.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 2000,
                    "limit": 1200
                }
            }
        ]).to_list(length=1200)
        
        if not similar_products:
            return []
        similar_products.sort(key=itemgetter("score"), reverse=True)

        product_nums = list(OrderedDict.fromkeys(p["product_num"] for p in similar_products))
        product_nums = product_nums[:100]

        latest_prices = await db.prices.aggregate([
            {"$match": {"product_num": {"$in": product_nums}}},
            {"$sort": {"date": -1}},
            {"$group": {
                "_id": {"product_num": "$product_num", "store_num": "$store_num"},
                "latest_price": {"$first": "$amount"},
                "latest_date": {"$first": "$date"},
                "unit": {"$first": "$unit"},
                "price_per_unit": {"$first": "$price_per_unit"}
            }}
        ]).to_list(length=1000)

        store_nums = list({p["_id"]["store_num"] for p in latest_prices})
        stores = await db.stores.find({"store_num": {"$in": store_nums}}).to_list(length=1000)
        store_map = {s["store_num"]: s.get("store_name", "Unknown Store") for s in stores}

        product_details = await db.products.find(
            {"product_num": {"$in": product_nums}},
            {
                "product_num": 1,
                "product_name": 1,
                "product_brand": 1,
                "product_link": 1,
                "image_url": 1,
                "category_path": 1
            }
        ).to_list(length=2000)
        product_map = {p["product_num"]: p for p in product_details}

        price_flag_filter = {
            "$or": [
                {"product_num": p["_id"]["product_num"], "store_num": p["_id"]["store_num"]}
                for p in latest_prices
            ]
        }
        price_flags = await db.price_flags.find(price_flag_filter).to_list(length=1000)
        flag_map = {
            (pf["product_num"], pf["store_num"]): pf
            for pf in price_flags
        }

        response = []
        for price in latest_prices:
            pid = price["_id"]["product_num"]
            sid = price["_id"]["store_num"]
            product = product_map.get(pid, {})
            flags = flag_map.get((pid, sid), {})

            response.append({
                "product_num": pid,
                "store_num": sid,
                "store_name": store_map.get(sid, "Unknown Store"),
                "product_name": product.get("product_name", "Unknown Product"),
                "product_brand": product.get("product_brand", "Unknown Brand"),
                "product_link": product.get("product_link", ""),
                "image_url": product.get("image_url", ""),
                "category_path": product.get("category_path", []),
                "latest_price": price.get("latest_price"),
                "latest_date": price.get("latest_date"),
                "unit": price.get("unit"),
                "price_per_unit": price.get("price_per_unit"),
                "best_in_30d": flags.get("best_in_30d", False),
                "best_in_90d": flags.get("best_in_90d", False),
                "std_from_mean": flags.get("std_from_mean", None),
                "discount_percent": flags.get("discount_percent", None),
            })

        return response

    except Exception as e:
        print("❌ Semantic search failed:", str(e))
        raise HTTPException(status_code=500, detail="Internal server error during semantic search.")



@router.get("/products/GetProduct")
async def get_product(search: str = Query(...)):
    products_cursor = db.products.find({"search_terms": search.lower()})
    products = await products_cursor.to_list(length=600)
    product_map = {}
    for p in products:
        if p["product_num"] not in product_map:
            product_map[p["product_num"]] = p
        if len(product_map) >= 100:
            break
    products = list(product_map.values())

    if not products:
        return []

    product_nums = [p["product_num"] for p in products]

    latest_prices = await db.prices.aggregate([
        {"$match": {"product_num": {"$in": product_nums}}},
        {"$sort": {"date": -1}},
        {"$group": {
            "_id": {"product_num": "$product_num", "store_num": "$store_num"},
            "latest_price": {"$first": "$amount"},
            "latest_date": {"$first": "$date"},
            "unit": {"$first": "$unit"},
            "price_per_unit": {"$first": "$price_per_unit"}
        }}
    ]).to_list(length=1000)

    store_nums = list({p["_id"]["store_num"] for p in latest_prices})
    stores_cursor = db.stores.find({"store_num": {"$in": store_nums}})
    stores = await stores_cursor.to_list(length=2000)
    store_map = {s["store_num"]: s["store_name"] for s in stores}

    product_details_cursor = db.products.find(
        {"product_num": {"$in": product_nums}},
        {"product_num": 1, "product_name": 1, "product_brand": 1,
         "product_link": 1, "image_url": 1, "category_path": 1}
    )
    product_details = await product_details_cursor.to_list(length=2000)
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
            "category_path": product.get("category_path", []),
            "latest_price": price["latest_price"],
            "latest_date": price["latest_date"],
            "unit": price["unit"],
            "price_per_unit": price.get("price_per_unit")
        })
    return response

@router.get("/products/GetDeals")
async def get_deals():
    try:
        # Step 1: Get all price flags sorted by discount
        top_flags = await db.price_flags.find(
            {"discount_percent": {"$ne": None}}
        ).sort("discount_percent", -1).limit(100).to_list(length=100)

        if not top_flags:
            return []

        combos = [(pf["product_num"], pf["store_num"]) for pf in top_flags]
        product_nums = [p for p, _ in combos]
        store_nums = [s for _, s in combos]

        # Step 2: Product details
        products = await db.products.find(
            {"product_num": {"$in": product_nums}},
            {"product_num": 1, "product_name": 1, "product_brand": 1,
             "product_link": 1, "image_url": 1, "category_path": 1}
        ).to_list(length=100)
        product_map = {p["product_num"]: p for p in products}

        # Step 3: Store info
        stores = await db.stores.find({"store_num": {"$in": store_nums}}).to_list(length=100)
        store_map = {s["store_num"]: s.get("store_name", "Unknown Store") for s in stores}

        # Step 4: Get latest prices
        latest_prices = await db.prices.aggregate([
            {"$match": {"product_num": {"$in": product_nums}}},
            {"$sort": {"date": -1}},
            {"$group": {
                "_id": {"product_num": "$product_num", "store_num": "$store_num"},
                "latest_price": {"$first": "$amount"},
                "latest_date": {"$first": "$date"},
                "unit": {"$first": "$unit"}
            }}
        ]).to_list(length=2000)

        price_map = {
            (p["_id"]["product_num"], p["_id"]["store_num"]): p for p in latest_prices
        }

        # Step 5: Final assemble
        deals = []
        for pf in top_flags:
            pid, sid = pf["product_num"], pf["store_num"]
            product = product_map.get(pid, {})
            store = store_map.get(sid, "Unknown Store")
            price = price_map.get((pid, sid), {})

            deals.append({
                "product_num": pid,
                "store_num": sid,
                "store_name": store,
                "product_name": product.get("product_name", ""),
                "product_brand": product.get("product_brand", ""),
                "product_link": product.get("product_link", ""),
                "image_url": product.get("image_url", ""),
                "category_path": product.get("category_path", []),
                "latest_price": price.get("latest_price"),
                "latest_date": price.get("latest_date"),
                "unit": price.get("unit"),
                "discount_percent": pf["discount_percent"]
            })

        return deals

    except Exception as e:
        print("❌ GetDeals failed:", str(e))
        raise HTTPException(status_code=500, detail="Failed to get top deals.")
