from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = Path(__file__).parent.parent / "data"


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


def _row_to_dict(row: pd.Series) -> dict:
    return {
        "movieId": int(row["movieId"]),
        "title": row["title"],
        "genres": row["genres"],
    }


def get_recommendations(movie_id: int, top_n: int = 10) -> list[dict]:
    if movie_id not in _id_to_row:
        raise KeyError(f"movie_id {movie_id} not found")

    idx = _id_to_row[movie_id]
    scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

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
