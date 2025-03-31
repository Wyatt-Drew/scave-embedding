from fastapi import APIRouter
from db import get_db
from datetime import datetime, timedelta

router = APIRouter()
db = get_db()

@router.get("/baskets/GetBasketPrices")
async def get_basket_prices():
    today = datetime.utcnow()
    start_date = today - timedelta(weeks=15)

    basket_prices = await db.basket_prices.aggregate([
        {"$match": {"date": {"$gte": start_date, "$lte": today}}},
        {"$group": {
            "_id": {"store_num": "$store_num", "date": "$date"},
            "basket_price": {"$first": "$basket_price"}
        }},
        {"$project": {
            "_id": 0,
            "store_num": "$_id.store_num",
            "date": "$_id.date",
            "basket_price": 1
        }},
        {"$sort": {"store_num": 1, "date": -1}}
    ]).to_list(length=100)

    return basket_prices