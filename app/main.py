from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.recommender import get_recommendations, search_titles

app = FastAPI(title="Video Recommender")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/api/search")
def search(q: str = "", limit: int = 10):
    return search_titles(q, limit)


@app.get("/api/recommend/{movie_id}")
def recommend(movie_id: int, top_n: int = 10):
    try:
        return get_recommendations(movie_id, top_n)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
