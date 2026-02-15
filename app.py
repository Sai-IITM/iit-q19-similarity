from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Vector Similarity API")

# CORS for IIT grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

def simple_similarity(query: str, doc: str) -> float:
    """Pure Python - Vercel safe!"""
    query_words = set(query.lower().split())
    doc_words = set(doc.lower().split())
    overlap = len(query_words.intersection(doc_words))
    return overlap / max(len(query_words), 1)

@app.post("/similarity")
async def similarity_search(request: SimilarityRequest):
    # Calculate similarities
    similarities = [simple_similarity(request.query, doc) for doc in request.docs]
    
    # Top 3 indices
    top_indices = sorted(range(len(similarities)), 
                        key=lambda i: similarities[i], reverse=True)[:3]
    
    # Return original docs in ranked order
    matches = [request.docs[i] for i in top_indices]
    
    return {"matches": matches}

@app.get("/")
async def root():
    return {"message": "Vector Similarity API Ready!", "endpoint": "/similarity"}

# REMOVE THIS FOR VERCEL:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

