"""
Streamlit app for semantic movie search.
"""
import streamlit as st
from utils import search_similar_text

st.set_page_config(
    page_title="Movie Search",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Search")
st.markdown("Describe a movie and find similar ones based on plot synopses.")

with st.form("search_form"):
    query = st.text_area(
        "What kind of movie are you looking for?",
        placeholder="e.g. A dark comedy about a hitman who finds redemption...",
        height=100,
    )
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    if not query.strip():
        st.warning("Please enter a description.")
    else:
        with st.spinner("Searching..."):
            results = search_similar_text(query.strip())

        if isinstance(results, str):
            st.error(results)
        elif not results:
            st.info("No results found.")
        else:
            st.success(f"Found {len(results)} similar movies")
            for doc in results:
                score = doc.get("score", 0)
                imdb_id = doc.get("imdb_id")
                imdb_url = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else ""
                header = f"🎬 {doc.get('title', 'Unknown')} - Score: {score:.4f}"
                if imdb_url:
                    header += f" - {imdb_url}"
                with st.expander(header):
                    if doc.get("plot_synopsis"):
                        st.write(doc["plot_synopsis"])