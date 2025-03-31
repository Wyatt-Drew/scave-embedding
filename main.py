from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.products import router as products_router
from routes.prices import router as prices_router
from routes.baskets import router as baskets_router
from db import connect_to_mongo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db():
    await connect_to_mongo()

app.include_router(products_router, prefix="/api")
app.include_router(prices_router, prefix="/api")
app.include_router(baskets_router, prefix="/api")