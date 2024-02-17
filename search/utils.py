"""
utility functions to perform search over a milvus index
"""
from wtforms import Form, validators, TextAreaField
from wtforms.widgets import TextArea
import requests
from pymilvus import (
    connections,
    utility,
    Collection,
)

class InputForm(Form):
    Text = TextAreaField('', widget=TextArea(),validators=[validators.DataRequired()])

EMBEDDINGS_ENDPOINT = 'http://localhost:8080/embeddings'
COLLECTION_NAME = 'mspt_movies'
SEARCH_PARAMS = {
    "metric_type": "L2",
    "params": {"nprobe": 10}, 
}

def vectorize_text(text):
    "returns vectors for a text"    
    myobj = {"text":text}
    x = requests.post(EMBEDDINGS_ENDPOINT, json = myobj, timeout=1.5)
    return x.json()['emb_vector']

def search_similar_text(text):
    "based on text provided returns similar docs"
    #exposed to the machine with 8080
    connections.connect("default", host="milvus-standalone", port=19530)
    has = utility.has_collection(COLLECTION_NAME)
    if not has:
        res = f"The is no collection with name:{COLLECTION_NAME}"
    else:
        search_text_vector = vectorize_text(text)
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
                res.append({'id': hit.id,
                            'title': hit.entity.get('title'),
                            'score': hit.score}) 
    return res
