# Video Recommender

A content-based recommendation demo built on the MovieLens dataset (`data/`), treating movies as
stand-in "videos". Recommendations are computed from TF-IDF vectors over each title's genres and
user-supplied tags, ranked by cosine similarity.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/ in a browser. Search for a title, click a suggestion, and see similar
titles ranked by shared genres/tags.

## API

- `GET /api/search?q=toy&limit=10` - title search for autocomplete.
- `GET /api/recommend/{movie_id}?top_n=10` - top-N similar titles by content similarity.

Interactive docs available at http://127.0.0.1:8000/docs.

## Notes

- The TF-IDF matrix is built once in memory at process startup (`app/recommender.py`) — at this
  dataset size (~9.7k titles) this takes well under a second, so no model artifact is persisted.
  `models/` is kept as a placeholder for where a serialized model would go if the dataset grew
  large enough to make on-demand fitting too slow.
- No database or auth — data is read directly from the CSVs in `data/` on startup.
