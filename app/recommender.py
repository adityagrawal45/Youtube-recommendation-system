from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.collaborative import build_item_similarity, load_ratings

DATA_DIR = Path(__file__).parent.parent / "data"

CONTENT_WEIGHT = 0.5
CF_WEIGHT = 0.5
MIN_RATINGS_FOR_CF = 5


def load_data() -> pd.DataFrame:
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")

    movies["genres_text"] = (
        movies["genres"]
        .replace("(no genres listed)", "")
        .str.replace("|", " ", regex=False)
    )

    tags_per_movie = (
        tags.groupby("movieId")["tag"]
        .apply(lambda vals: " ".join(str(v).lower() for v in vals))
        .rename("tags_text")
    )

    movies = movies.merge(tags_per_movie, on="movieId", how="left")
    movies["tags_text"] = movies["tags_text"].fillna("")

    movies["content"] = (movies["genres_text"] + " ") * 2 + movies["tags_text"]

    return movies


movies_df = load_data()
_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = _vectorizer.fit_transform(movies_df["content"])

_id_to_row = pd.Series(movies_df.index, index=movies_df["movieId"])

_ratings_df = load_ratings()
_item_similarity, _item_rating_counts = build_item_similarity(_ratings_df, movies_df["movieId"])


def _row_to_dict(row: pd.Series) -> dict:
    return {
        "movieId": int(row["movieId"]),
        "title": row["title"],
        "genres": row["genres"],
    }


def _normalize(arr: np.ndarray) -> np.ndarray:
    low, high = arr.min(), arr.max()
    if high - low == 0:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def get_recommendations(movie_id: int, top_n: int = 10, method: str = "hybrid") -> list[dict]:
    if method not in {"content", "collaborative", "hybrid"}:
        raise ValueError(f"unknown method {method!r}")

    if movie_id not in _id_to_row:
        raise KeyError(f"movie_id {movie_id} not found")

    idx = _id_to_row[movie_id]
    content_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    has_enough_ratings = _item_rating_counts[idx] >= MIN_RATINGS_FOR_CF

    if method == "content" or not has_enough_ratings:
        scores = content_scores
    else:
        cf_scores = np.asarray(_item_similarity[idx].todense()).flatten()
        if method == "collaborative":
            scores = cf_scores
        else:
            scores = CONTENT_WEIGHT * _normalize(content_scores) + CF_WEIGHT * _normalize(cf_scores)

    ranked = scores.argsort()[::-1]
    ranked = [i for i in ranked if i != idx][:top_n]

    return [_row_to_dict(movies_df.iloc[i]) for i in ranked]


def search_titles(query: str, limit: int = 10) -> list[dict]:
    query = query.strip().lower()
    if not query:
        return []

    matches = movies_df[movies_df["title"].str.lower().str.contains(query, regex=False)]
    matches = matches.sort_values("title").head(limit)

    return [_row_to_dict(row) for _, row in matches.iterrows()]
