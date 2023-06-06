from fastapi import FastAPI 
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
app = FastAPI()

class Item(BaseModel):
    text: str = ""

@app.post("/embeddings")
async def t5(input: Item):
    text_embedding = model.encode(input.text).tolist()
    return {"emb_vector": text_embedding}