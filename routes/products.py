from fastapi import APIRouter, Query, HTTPException
from db import get_db
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import re

router = APIRouter()
db = get_db()
model = SentenceTransformer("all-MiniLM-L6-v2")


from fastapi import HTTPException, Query
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@router.get("/products/SemanticSearch")
async def semantic_search(query: str = Query(...)):
    query_vector = model.encode(query).tolist()
    q_vec_np = np.array(query_vector).reshape(1, -1)

    try:
        # Step 1: Vector Search
        similar_products_raw = await db.products.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 2000,
                    "limit": 100
                }
            },
            {
                "$project": {
                    "product_num": 1,
                    "product_name": 1,
                    "search_terms": 1,
                    "embedding": 1
                }
            }
        ]).to_list(length=100)

        if not similar_products_raw:
            return []

        # Tokenize query for keyword overlap
        query_tokens = set(re.findall(r'\w+', query.lower()))

        # Compute cosine similarity + keyword overlap hybrid score
        for doc in similar_products_raw:
            emb = np.array(doc["embedding"]).reshape(1, -1)
            cosine_sim = cosine_similarity(q_vec_np, emb)[0][0]
            doc["cosine_similarity"] = cosine_sim

            name_tokens = set(re.findall(r'\w+', doc.get("product_name", "").lower()))
            terms_tokens = set(doc.get("search_terms", []))
            overlap_count = len(query_tokens & (name_tokens | terms_tokens))

            doc["score"] = 0.85 * cosine_sim + 0.15 * (overlap_count / max(1, len(query_tokens)))

        # Sort by hybrid score
        filtered_products = sorted(similar_products_raw, key=lambda d: d["score"], reverse=True)
        top_docs = filtered_products[:30]
        product_nums = [p["product_num"] for p in top_docs]

        # Step 2: Latest Prices
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
        ).to_list(length=100)
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

        # Step 3: Assemble and sort final response
        response = []
        for price in latest_prices:
            pid = price["_id"]["product_num"]
            sid = price["_id"]["store_num"]
            product = product_map.get(pid, {})
            flags = flag_map.get((pid, sid), {})

            doc = next((d for d in top_docs if d["product_num"] == pid), {})
            score = doc.get("score", 0)

            response.append({
                "score": score,  # temporary for sorting
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
                "discount_percent": flags.get("discount_percent", None)
            })

        # Final sort by score
        sorted_response = sorted(response, key=lambda x: x["score"], reverse=True)
        for r in sorted_response:
            r.pop("score", None)

        return sorted_response

    except Exception as e:
        print("❌ Semantic search failed:", str(e))
        raise HTTPException(status_code=500, detail="Internal server error during semantic search.")




@router.get("/products/GetProduct")
async def get_product(search: str = Query(...)):
    # Tokenize input: lowercase, remove non-word characters, split into keywords
    tokens = re.findall(r'\w+', search.lower())

    if not tokens:
        return []

    # Mongo query: match if ANY search_term contains ANY of the tokens
    query = {
        "search_terms": {
            "$in": tokens
        }
    }

    # Fetch matching products
    products_cursor = db.products.find(query)
    products = await products_cursor.to_list(length=600)

    if not products:
        return []

    product_nums = [p["product_num"] for p in products]

    # Step 2: Fetch latest prices
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
    ]).to_list(length=100)

    # Step 3: Store details
    store_nums = list({p["_id"]["store_num"] for p in latest_prices})
    stores_cursor = db.stores.find({"store_num": {"$in": store_nums}})
    stores = await stores_cursor.to_list(length=2000)
    store_map = {s["store_num"]: s["store_name"] for s in stores}

    # Step 4: Product details
    product_details_cursor = db.products.find(
        {"product_num": {"$in": product_nums}},
        {"product_num": 1, "product_name": 1, "product_brand": 1,
         "product_link": 1, "image_url": 1, "category_path": 1}
    )
    product_details = await product_details_cursor.to_list(length=2000)
    product_map = {p["product_num"]: p for p in product_details}

    # Step 5: Construct response
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
        all_flags = await db.price_flags.find(
            {"discount_percent": {"$ne": None}}
        ).sort("discount_percent", -1).limit(500).to_list(length=500)

        if not all_flags:
            return []

        # Step 2: Get latest prices for all flagged items
        flagged_combos = [(pf["product_num"], pf["store_num"]) for pf in all_flags]
        product_nums = [pid for pid, _ in flagged_combos]

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

        latest_price_map = {
            (p["_id"]["product_num"], p["_id"]["store_num"]): p
            for p in latest_prices
        }

        # Step 3: For each product_num, choose the flag with the lowest latest_price
        best_flag_map = {}
        for pf in all_flags:
            pid, sid = pf["product_num"], pf["store_num"]
            price_info = latest_price_map.get((pid, sid))
            if not price_info:
                continue

            latest_price = price_info["latest_price"]
            if pid not in best_flag_map or latest_price < best_flag_map[pid]["latest_price"]:
                pf_copy = pf.copy()
                pf_copy["latest_price"] = latest_price
                pf_copy["latest_date"] = price_info["latest_date"]
                pf_copy["unit"] = price_info.get("unit")
                best_flag_map[pid] = pf_copy

        # Final sorted flags by discount_percent
        top_flags = sorted(best_flag_map.values(), key=lambda x: x["discount_percent"], reverse=True)

        combos = [(pf["product_num"], pf["store_num"]) for pf in top_flags]
        product_nums = [p for p, _ in combos]
        store_nums = [s for _, s in combos]

        # Step 4: Product and store info
        products = await db.products.find(
            {"product_num": {"$in": product_nums}},
            {
                "product_num": 1,
                "product_name": 1,
                "product_brand": 1,
                "product_link": 1,
                "image_url": 1,
                "category_path": 1
            }
        ).to_list(length=len(product_nums))
        product_map = {p["product_num"]: p for p in products}

        stores = await db.stores.find({"store_num": {"$in": store_nums}}).to_list(length=len(store_nums))
        store_map = {s["store_num"]: s.get("store_name", "Unknown Store") for s in stores}

        # Step 5: Assemble response
        deals = []
        for pf in top_flags:
            pid, sid = pf["product_num"], pf["store_num"]
            product = product_map.get(pid)
            if not product:
                continue

            deals.append({
                "product_num": pid,
                "store_num": sid,
                "store_name": store_map.get(sid, "Unknown Store"),
                "product_name": product.get("product_name", ""),
                "product_brand": product.get("product_brand", ""),
                "product_link": product.get("product_link", ""),
                "image_url": product.get("image_url", ""),
                "category_path": product.get("category_path", []),
                "latest_price": pf.get("latest_price"),
                "latest_date": pf.get("latest_date"),
                "unit": pf.get("unit"),
                "discount_percent": pf["discount_percent"]
            })

        return deals

    except Exception as e:
        print("❌ GetDeals failed:", str(e))
        raise HTTPException(status_code=500, detail="Failed to get top deals.")
