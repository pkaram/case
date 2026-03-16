"""
Vector search over Qdrant (semantic similarity only).
"""
import requests
from qdrant_client import QdrantClient, models

EMBEDDINGS_ENDPOINT = "http://embeddings_gen:80/embeddings"
COLLECTION_NAME = "mspt_movies"
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333


def vectorize_text(text):
    "returns dense vectors for a text (from our embeddings API)"
    myobj = {"text": text}
    x = requests.post(EMBEDDINGS_ENDPOINT, json=myobj, timeout=60)
    return x.json()["emb_vector"]


def search_similar_text(text):
    "vector search: semantic similarity over dense embeddings"
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(COLLECTION_NAME):
        return f"No collection with name: {COLLECTION_NAME}. Run: python vectorizer/vectorize_items.py -k 100"

    dense_vector = vectorize_text(text)

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vector,
        limit=10,
    )

    points = [
        {
            "id": point.id,
            "title": point.payload.get("title"),
            "plot_synopsis": point.payload.get("plot_synopsis"),
            "imdb_id": point.payload.get("imdb_id"),
            "score": point.score,
        }
        for point in result.points
    ]
    # Euclidean: lower score = more similar (SMALLER_IS_BETTER)
    return sorted(points, key=lambda p: p["score"], reverse=False)
