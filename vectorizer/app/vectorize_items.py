#Link to download data from and store it in folder data
#https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download

import requests
import time
import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

DIM=768

def sleep_for(n):
    time.sleep(n)

def vectorize_text(text):
    sleep_for(1)
    url = 'http://embeddings_gen:8080/embeddings'
    myobj = {"text":text}
    x = requests.post(url, json = myobj)
    return x.json()['emb_vector']

def index_to_milvus(df):
    #establish connection to milvus
    collection_name = "mspt_movies"
    connections.connect("default", host="milvus-standalone", port="19530") #exposed to the machine with 8080
    has = utility.has_collection(collection_name)
    if has:
        print(f"Does collection mspt_movies_embs exist in Milvus: {has}")
        utility.drop_collection(collection_name)
    fields = [
    FieldSchema(name="imdb_id", dtype=DataType.STRING, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="title", dtype=DataType.STRING, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    ]
    schema = CollectionSchema(fields, "mspt_movies embeddings")
    embs = Collection(collection_name, schema, consistency_level="Strong", auto_id=False)
    #milvus takes data in list as data in fields 
    entities = [
    # provide the pk field because `auto_id` is set to False
    df.imdb_id.values.tolist(),
    df.title.values.tolist(),  # field random, only supports list
    df.EMB.values.tolist(),    # field embeddings, supports numpy.ndarray and list
    ]

    insert_result = embs.insert(entities)
    print(f"Number of inserts in Milvus: {insert_result.insert_count}")  # check the num_entites

def main():
    sleep_for(120)
    data = pd.read_csv('mpst_full_data.csv')
    data_ = data.head(120) #keep 100 samples to showcase
    #data_['EMB'] = data_.apply(lambda x: vectorize_text(x.plot_synopsis), axis=1)
    data_['EMB'] = None
    for i in range(len(data_)):
        data_['EMB'][i] = vectorize_text(data_.plot_synopsis[i])
    index_to_milvus(data_)


if __name__ == '__main__':
    main()
