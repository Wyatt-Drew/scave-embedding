from fastapi import APIRouter, Query
from db import get_db
from datetime import datetime, timedelta

router = APIRouter()
db = get_db()

@router.get("/prices/GetProductHistory")
async def get_product_history(product_num: str = Query(...)):
    today = datetime.utcnow()
    start_date = today - timedelta(weeks=15)

    price_history = await db.prices.aggregate([
        {"$match": {
            "product_num": product_num,
            "date": {"$gte": start_date, "$lte": today}
        }},
        {"$sort": {"store_num": 1, "date": -1}},
        {"$group": {
            "_id": {"store_num": "$store_num", "date": "$date"},
            "amount": {"$first": "$amount"},
            "price_per_unit": {"$first": "$price_per_unit"},
            "unit": {"$first": "$unit"}
        }},
        {"$project": {
            "_id": 0,
            "store_num": "$_id.store_num",
            "date": "$_id.date",
            "amount": 1,
            "price_per_unit": 1,
            "unit": 1
        }},
        {"$sort": {"store_num": 1, "date": -1}}
    ]).to_list(length=100)

    return price_history