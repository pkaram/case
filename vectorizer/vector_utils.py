"""
vector utils functions
"""
import requests

ENDPOINT = 'http://localhost:8080/embeddings'

def vectorize_text(text):
    "returns vectors for a text"
    myobj = {"text":text}
    x = requests.post(ENDPOINT, json = myobj, timeout=1.5)
    return x.json()['emb_vector']
