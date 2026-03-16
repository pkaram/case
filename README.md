# case

Context Aware Search Engine.

This repo uses sentence embeddings with Qdrant (a vector database) to build a search engine. Requires conda and Docker Compose.

1. **Dataset**: Movie plot synopses from [Kaggle](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download). Download the CSV and place it in the `vectorizer` folder.

2. **Start services** (Qdrant, embeddings API, search UI):
   ```
   docker compose up -d
   ```

3. **Create conda environment**:
   ```
   conda create -n search python=3.13 -y
   conda activate search
   pip install -r vectorizer/requirements.txt
   ```

4. **Populate Qdrant** with embeddings:
   ```
   python vectorizer/vectorize_items.py -k 100 #to index 100 movies
   # remove flag to index all
   ```

5. **Search**:
   - Open http://localhost:5001/ in your browser (or `SEARCH_PORT=5000` if port 5000 is free)
   - Or: `python vectorizer/search_item.py -m "your movie description"`
