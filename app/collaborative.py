from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ratings.csv")


def build_item_similarity(ratings: pd.DataFrame, movie_ids: pd.Series):
    """Build an item-item cosine similarity matrix aligned to movie_ids order.

    Returns (item_similarity, item_rating_counts), where item_similarity is a
    sparse movies x movies matrix and item_rating_counts is an array of how
    many users rated each movie (same order as movie_ids).
    """
    movie_id_to_pos = {mid: i for i, mid in enumerate(movie_ids)}
    ratings = ratings[ratings["movieId"].isin(movie_id_to_pos)]

    user_cat = ratings["userId"].astype("category")
    user_idx = user_cat.cat.codes.to_numpy()
    movie_idx = ratings["movieId"].map(movie_id_to_pos).to_numpy()

    n_users = user_cat.cat.categories.size
    n_movies = len(movie_ids)

    ui_matrix = csr_matrix(
        (ratings["rating"].to_numpy(), (user_idx, movie_idx)),
        shape=(n_users, n_movies),
    )

    item_similarity = cosine_similarity(ui_matrix.T, dense_output=False)
    item_rating_counts = np.asarray((ui_matrix != 0).sum(axis=0)).flatten()

    return item_similarity, item_rating_counts
