"""
It produces 

Link to download data from and store it in folder:
https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download
"""
import argparse
import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from vector_utils import vectorize_text

DIM=768

def index_to_milvus(df):
    collection_name = "mspt_movies"
    connections.connect("default", host="localhost", port=19530)
    has = utility.has_collection(collection_name)
    if has:
        print(f"Does collection mspt_movies_embs exist in Milvus: {has} and will be dropped")
        utility.drop_collection(collection_name)
    fields = [
    FieldSchema(name="imdb_id", dtype=DataType.VARCHAR,
    is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="title", dtype=DataType.VARCHAR, auto_id=False, max_length=100),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    ]
    schema = CollectionSchema(fields, "mspt_movies embeddings")
    embs = Collection(collection_name, schema, consistency_level="Strong", auto_id=False)
    entities = [
    df.imdb_id.values.tolist(),
    df.title.values.tolist(),
    df.EMB.values.tolist(),
    ]
    insert_result = embs.insert(entities)
    print(f"Number of inserts in Milvus: {insert_result.insert_count}")
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 768},
    }
    embs.create_index("embeddings", index)
    print("index created")

def main(arguments):
    data = pd.read_csv('vectorizer/mpst_full_data.csv')
    if str(arguments.items_count) == 0:
        data_ = data
    else:
        data_ = data.head(int(args.items_count))
    data_['EMB'] = None
    for i in range(len(data_)):
        data_['EMB'][i] = vectorize_text(data_.plot_synopsis[i])
    index_to_milvus(data_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--items_count", help="Description of the movie")
    args = parser.parse_args()
    main(args)
