import pytest

from app.recommender import _item_rating_counts, _id_to_row, get_recommendations

TOY_STORY_ID = 1


def test_hybrid_returns_top_n_excluding_query_movie():
    results = get_recommendations(TOY_STORY_ID, top_n=10, method="hybrid")
    assert len(results) == 10
    assert all(r["movieId"] != TOY_STORY_ID for r in results)


@pytest.mark.parametrize("method", ["content", "collaborative", "hybrid"])
def test_each_method_returns_top_n_excluding_query_movie(method):
    results = get_recommendations(TOY_STORY_ID, top_n=5, method=method)
    assert len(results) == 5
    assert all(r["movieId"] != TOY_STORY_ID for r in results)


def test_unknown_movie_id_raises_key_error():
    with pytest.raises(KeyError):
        get_recommendations(999_999_999, top_n=5)


def test_unknown_method_raises_value_error():
    with pytest.raises(ValueError):
        get_recommendations(TOY_STORY_ID, top_n=5, method="bogus")


def test_low_rating_count_movie_falls_back_to_content_only():
    low_rated_id = next(
        movie_id
        for movie_id, idx in _id_to_row.items()
        if _item_rating_counts[idx] < 5
    )

    content_result = get_recommendations(low_rated_id, top_n=5, method="content")
    hybrid_result = get_recommendations(low_rated_id, top_n=5, method="hybrid")

    assert hybrid_result == content_result
