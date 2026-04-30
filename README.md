# Song2Vec Recommender System Using Word2Vec Embeddings

This project implements a Word2Vec-based music recommendation system using playlist sequences from the 30Music dataset. Songs are treated as tokens and playlists as sentences to learn embedding-based song similarities.

## Technologies
- Python
- Jupyter Notebook
- Pandas, NumPy
- Gensim Word2Vec
- Matplotlib / Seaborn
- t-SNE
- HR@K and NDCG@K evaluation

## Project Highlights
- Preprocessed raw playlist and track metadata into normalized artist-track sequences.
- Trained and optimized Skip-gram Word2Vec models for next-song prediction.
- Evaluated recommendation quality using HR@K and NDCG@K.
- Analyzed long-tail popularity, cold-start/OOV behavior, and embedding neighborhoods.
- Implemented an artist-level fallback strategy for out-of-vocabulary songs.
