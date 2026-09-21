const searchInput = document.getElementById("search");
const suggestionsEl = document.getElementById("suggestions");
const resultsEl = document.getElementById("results");
const resultsHeadingEl = document.getElementById("results-heading");

let debounceTimer = null;

searchInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  const query = searchInput.value.trim();

  if (!query) {
    suggestionsEl.innerHTML = "";
    return;
  }

  debounceTimer = setTimeout(() => runSearch(query), 250);
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-box")) {
    suggestionsEl.innerHTML = "";
  }
});

async function runSearch(query) {
  const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  const items = await res.json();
  renderSuggestions(items);
}

function renderSuggestions(items) {
  suggestionsEl.innerHTML = "";

  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No matches found.";
    suggestionsEl.appendChild(empty);
    return;
  }

  for (const item of items) {
    const row = document.createElement("div");
    row.className = "suggestion-item";

    const title = document.createElement("div");
    title.textContent = item.title;

    const genres = document.createElement("div");
    genres.className = "genres";
    genres.textContent = item.genres;

    row.appendChild(title);
    row.appendChild(genres);
    row.addEventListener("click", () => selectMovie(item));

    suggestionsEl.appendChild(row);
  }
}

async function selectMovie(item) {
  searchInput.value = item.title;
  suggestionsEl.innerHTML = "";

  const res = await fetch(`/api/recommend/${item.movieId}`);
  const recommendations = await res.json();
  renderResults(item, recommendations);
}

function renderResults(item, recommendations) {
  resultsHeadingEl.hidden = false;
  resultsHeadingEl.textContent = `Because you liked "${item.title}"`;
  resultsEl.innerHTML = "";

  if (recommendations.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No recommendations found.";
    resultsEl.appendChild(empty);
    return;
  }

  for (const rec of recommendations) {
    const card = document.createElement("div");
    card.className = "result-card";

    const title = document.createElement("div");
    title.textContent = rec.title;

    const genres = document.createElement("div");
    genres.className = "genres";
    genres.textContent = rec.genres;

    card.appendChild(title);
    card.appendChild(genres);
    resultsEl.appendChild(card);
  }
}
