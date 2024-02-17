"script to search for the most relevant docs in relation to text input"
import argparse
from pymilvus import (
    connections,
    utility,
    Collection,
)
from vector_utils import vectorize_text

COLLECTION_NAME = 'mspt_movies'
MOVIE_DESCRIPTION = 'A crime movie'
SEARCH_PARAMS = {
    "metric_type": "L2",
    "params": {"nprobe": 10}, 
}

def main(arguments):
    "returns most relevant docs in relation to text input"
    connections.connect("default", host="localhost", port="19530") #exposed to the machine with 8080
    has = utility.has_collection(COLLECTION_NAME)
    if not has:
        print(f"The is no collection with name:{COLLECTION_NAME}")
    else:
        search_text_vector = vectorize_text(arguments.movie_description)
        collection = Collection(COLLECTION_NAME)
        collection.load()

        results = collection.search(
            data=[search_text_vector],
            anns_field="embeddings",
            # the sum of `offset` in `param` and `limit`
            # should be less than 16384.
            param=SEARCH_PARAMS,
            limit=10,
            expr=None,
            # set the names of the fields you want to
            # retrieve from the search result.
            output_fields=['title'],
            consistency_level="Strong"
        )

        res = []
        for hits in results:
            for hit in hits:
                res.append([hit.id, hit.entity.get('title') ,hit.score])
                print(f"id: {hit.id}, title: {hit.entity.get('title')}, score {hit.score}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--movie_description", help="Description of the movie")
    args = parser.parse_args()
    main(args)
