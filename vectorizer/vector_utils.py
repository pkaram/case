"""
vector utils functions
"""
import requests
from tqdm import tqdm

ENDPOINT = "http://localhost:8080/embeddings"


def vectorize_text(text):
    "returns vectors for a single text"
    myobj = {"text": text}
    x = requests.post(ENDPOINT, json=myobj, timeout=60)
    return x.json()["emb_vector"]


def vectorize_texts_batch(texts, batch_size=32, timeout=300):
    "returns list of vectors for multiple texts (much faster than one-by-one)"
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{ENDPOINT}/batch",
            json={"texts": batch},
            timeout=timeout,
        )
        all_embeddings.extend(resp.json()["emb_vectors"])
    return all_embeddings
