# case

Context Aware Search Engine.

This repo takes advantage of sentence embeddings to showcase how a search engine can be created by using Milvus (a vector database). To run this, have conda and docker compose installed (if not already).

1. To run an example of a search engine, following movie dataset with plot synopses is used:
https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download
Download the csv file and store within *vectorizer* folder.

2. Following will create within the same network a milvus instance, an embeddings generator api and a search ui app.
```
docker-compose up -d
```

3. Create a conda environment and install requirements.
```
conda create -n search python=3.9 -y
conda activate search
pip install -r vectorizer/requirements.txt
```

3. To create and populate the milvus collection with embeddings for 100 movies run the following.
```
python vectorizer/vectorize_items.py -k 100
#set k=0 if embeddings for all movies should be created
python vectorizer/app/vectorize_items.py -k 0
```
Note that the more you increase the k parameter the more time it will be required for embeddings calculation.

4.  To search for a movie desciption open http://localhost:5000/ on your browser and give your text input 

    **OR** 

    You can use the python script with your description as argument as following
    ```
    python vectorizer/search_item.py -m "your text"
    ```
