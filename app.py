from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import numpy as np
from typing import List

app = FastAPI()

# CORS for IIT grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

def get_embedding(text: str):
    """OpenAI text-embedding-3-small"""
    response = client.embeddings.create(
        input=text[:8191],  # Token limit
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity formula"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
async def similarity_search(request: SimilarityRequest):
    # Generate embeddings
    query_embedding = get_embedding(request.query)
    doc_embeddings = [get_embedding(doc) for doc in request.docs]
    
    # Calculate cosine similarities
    similarities = [
        cosine_similarity(query_embedding, np.array(emb))
        for emb in doc_embeddings
    ]
    
    # Get top 3 most similar docs (by original index)
    top_indices = np.argsort(similarities)[-3:][::-1]
    
    # Return original document texts in ranked order
    matches = [request.docs[i] for i in top_indices]
    
    return {"matches": matches}

@app.get("/")
async def root():
    return {"message": "Vector Similarity API Ready! POST to /similarity"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
