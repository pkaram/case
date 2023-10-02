# sentence-embeddings

Dockerfile can used to create a standalone service to provide embeddings based on input text from pre-trained sentence-transformers.

Given an input text, it ouptuts a vector which captures the semantic information. The sentence vector may be used for information retrieval, clustering or sentence similarity tasks.

Model in use: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

Check performance of various models: https://www.sbert.net/docs/pretrained_models.html

By default, input text longer than 384 word pieces is truncated. Output list length is 768.


1.  Build image from Dockerfile

```
docker build -t sentence_emb .   
```

2. Create container from image

```
docker run -p 80:80 sentence_emb
```

3. Create embeddings for a text

```
curl -X POST "http://0.0.0.0:80/embeddings" -H "Content-Type: application/json" -d '{"text":"give the embeddings for this text"}'
```