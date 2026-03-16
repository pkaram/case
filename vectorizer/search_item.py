"script to search for the most relevant docs in relation to text input"
import argparse
from qdrant_client import QdrantClient
from vector_utils import vectorize_text

COLLECTION_NAME = "mspt_movies"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def main(arguments):
    "returns most relevant docs in relation to text input"
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"No collection with name: {COLLECTION_NAME}")
        return

    search_vector = vectorize_text(arguments.movie_description)
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=search_vector,
        limit=10,
    )

    for hit in result.points:
        print(f"id: {hit.id}, title: {hit.payload.get('title')}, score: {hit.score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--movie_description", help="Description of the movie")
    args = parser.parse_args()
    main(args)
