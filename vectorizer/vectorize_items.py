"""
Populates Qdrant with movie embeddings (dense vectors only).

Data source:
https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download
Download the csv file and store in the vectorizer folder.
"""
import argparse
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from vector_utils import vectorize_texts_batch

COLLECTION_NAME = "mspt_movies"
DENSE_DIM = 384  # all-MiniLM-L6-v2
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
UPSERT_BATCH_SIZE = 500


def index_to_qdrant(df):
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} exists, recreating...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.EUCLID),
    )

    points = [
        PointStruct(
            id=idx,
            vector=row["EMB"],
            payload={
                "imdb_id": row["imdb_id"],
                "title": row["title"],
                "plot_synopsis": row["plot_synopsis"],
            },
        )
        for idx, row in df.iterrows()
    ]

    for i in tqdm(range(0, len(points), UPSERT_BATCH_SIZE), desc="Indexing", unit="batch"):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
    print(f"Inserted {len(points)} points into Qdrant")


def main(arguments):
    data = pd.read_csv("vectorizer/mpst_full_data.csv")
    if arguments.items_count is None or arguments.items_count == 0:
        data_ = data
    else:
        data_ = data.head(arguments.items_count).copy()

    print(f"Indexing {len(data_)} movies...")
    texts = data_.plot_synopsis.tolist()
    embeddings = vectorize_texts_batch(texts, batch_size=32)
    data_["EMB"] = embeddings
    index_to_qdrant(data_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--items_count",
        type=int,
        default=None,
        help="Number of movies to index (default: all)",
    )
    args = parser.parse_args()
    main(args)
