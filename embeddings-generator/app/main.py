"an endpoint to provide embeddings based on given text"
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
app = FastAPI()

class Item(BaseModel):
    text: str = ""

class BatchItem(BaseModel):
    texts: list[str] = []

@app.post("/embeddings")
def embeddings_single(input: Item):
    """Single-text endpoint for backward compatibility."""
    text_embedding = model.encode(input.text).tolist()
    return {"emb_vector": text_embedding}

@app.post("/embeddings/batch")
def embeddings_batch(input: BatchItem):
    """Batch endpoint - encodes multiple texts in one call (much faster)."""
    if not input.texts:
        return {"emb_vectors": []}
    embeddings = model.encode(input.texts, show_progress_bar=False).tolist()
    return {"emb_vectors": embeddings}
