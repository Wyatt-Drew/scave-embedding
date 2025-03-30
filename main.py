from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Enable CORS so your Node backend can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Node backend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/vectorize")
async def vectorize_query(query: str = Query(..., min_length=1)):
    embedding = model.encode(query).tolist()
    return {"embedding": embedding}

@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping(request: Request):
    if request.method == "HEAD":
        return JSONResponse(status_code=200, content=None)
    return {"status": "ok"}