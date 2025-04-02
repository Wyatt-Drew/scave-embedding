from fastapi import APIRouter, Query
from db import get_db
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
import re

router = APIRouter()
db = get_db()
model = SentenceTransformer("all-MiniLM-L6-v2")

@router.get("/ping")
async def ping():
    return {"status": "ok"}

def cluster_and_label_products(products, max_clusters=8):
    product_texts = [p.get("product_name", "") + " " + p.get("description", "") for p in products]
    embeddings = model.encode(product_texts)

    if len(embeddings) < 3:
        return [{"category": "General", "products": products}]

    best_k = 2
    best_score = -1
    for k in range(2, min(max_clusters, len(embeddings)) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k

    # Final clustering with best_k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    final_labels = kmeans.fit_predict(embeddings)

    clustered = {}
    for label, product in zip(final_labels, products):
        clustered.setdefault(label, []).append(product)

    labeled_clusters = []
    for label, group in clustered.items():
        text = " ".join([p.get("product_name", "") for p in group])
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        common = Counter(words).most_common(3)
        category = " ".join([word for word, _ in common]) or "Other"

        labeled_clusters.append({
            "category": category.title(),
            "products": group
        })

    return labeled_clusters

@router.get("/products/SemanticSearch")
async def semantic_search(query: str = Query(...)):
    query_vector = model.encode(query).tolist()

    print("🔍 Incoming query:", query)
    print("🔢 Query vector (first 5 dims):", query_vector[:5])

    total_products = await db.products.count_documents({})
    with_embeddings = await db.products.count_documents({"embedding": {"$exists": True}})
    print(f"📦 Total products: {total_products}, with embeddings: {with_embeddings}")

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
        print(f"✅ Vector search returned {len(similar_products)} results.")
    except Exception as e:
        print("❌ Vector search failed:", str(e))
        return {"error": str(e)}

    if not similar_products:
        print("⚠️ No similar products found.")
        return []

    product_nums = [p["product_num"] for p in similar_products]
    print("🔗 Matching product_nums:", product_nums)

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
        {
            "product_num": 1,
            "product_name": 1,
            "product_brand": 1,
            "product_link": 1,
            "image_url": 1,
            "description": 1,
            "search_terms": 1
        }
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
            "search_terms": product.get("search_terms", []),
            "latest_price": price["latest_price"],
            "latest_date": price["latest_date"],
            "unit": price["unit"]
        })

    print(f"🧾 Final response size: {len(response)}")
    return response


@router.get("/products/SemanticSearchClustered")
async def semantic_search_clustered(query: str = Query(...)):
    query_vector = model.encode(query).tolist()

    try:
        similar_products = await db.products.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 300,
                    "limit": 50
                }
            }
        ]).to_list(length=50)

        if not similar_products:
            return []

        clustered = cluster_and_label_products(similar_products)
        return clustered

    except Exception as e:
        print("❌ Vector search error:", str(e))
        return {"error": str(e)}

@router.get("/products/GetProduct")
async def get_product(search: str = Query(...)):
    products_cursor = db.products.find({"search_terms": search.lower()})
    products = await products_cursor.to_list(length=100)

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
            "unit": {"$first": "$unit"}
        }}
    ]).to_list(length=100)

    store_nums = list({p["_id"]["store_num"] for p in latest_prices})
    stores_cursor = db.stores.find({"store_num": {"$in": store_nums}})
    stores = await stores_cursor.to_list(length=100)
    store_map = {s["store_num"]: s["store_name"] for s in stores}

    product_details_cursor = db.products.find(
        {"product_num": {"$in": product_nums}},
        {"product_num": 1, "product_name": 1, "product_brand": 1,
         "product_link": 1, "image_url": 1, "description": 1}
    )
    product_details = await product_details_cursor.to_list(length=100)
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
    return response
