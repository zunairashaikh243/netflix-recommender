# Netflix Content-Based Recommendation System

**Internship Task 1 (Easy)** - recommend similar Netflix titles from genres, categories and other content attributes.

## 1. Problem
Given a title a user liked, return the most similar titles in the catalogue. The dataset has no user ratings or watch history, so this is a **content-based** recommender (items are compared by their attributes, not by what other users watched).

## 2. Dataset
`Dataset.csv` - 8,790 rows, 10 columns: `show_id, type, title, director, country, date_added, release_year, rating, duration, listed_in`.

| Fact | Value |
|---|---|
| Movies / TV Shows | 6,126 / 2,664 |
| Distinct genre tags in `listed_in` | 42 |
| Missing values | stored as the text `"Not Given"` (2,588 directors, 287 countries) |
| Duplicate titles | 3 (dropped) |

There is no plot-description column, so genres/categories carry most of the signal.

## 3. Method (follows the 5 task steps)

**Step 1 - Prepare features.** Cleaned `"Not Given"` to empty, dropped duplicate titles, split comma-separated fields (`listed_in`, `director`, `country`) into tokens, and engineered three new features: *duration bucket* (short/medium/long movie, 1 / 2-3 / 4+ season show), *release era*, and *type*.

**Step 2 - Text to numbers.** One TF-IDF block per feature group (`TfidfVectorizer` with a comma tokenizer so "International Movies" stays one token). Blocks are multiplied by weights and stacked into one sparse matrix (8,787 titles x 5,141 features):

| Feature | Weight |
|---|---|
| Genres (`listed_in`) | 3.0 |
| Type (Movie / TV Show) | 1.5 |
| Country, Director | 1.0 each |
| Rating | 0.7 |
| Duration bucket, Era | 0.5 each |

**Step 3 - Similarity.** Rows are L2-normalised, so cosine similarity = dot product. Scores are computed per query (no 8,790 x 8,790 matrix stored).

**Step 4 - Recommend.** `recommend("Stranger Things", n=10)` returns the top-N titles with scores. Title lookup is case-insensitive with fuzzy matching for typos; `same_type_only=True` restricts to movies or shows.

**Step 5 - Evaluate.** No ratings exist, so offline proxy metrics are computed over 1,500 random query titles at K=10 and compared with a random baseline.

## 4. Sample recommendations

| Watched | Top recommendations |
|---|---|
| Stranger Things | Chilling Adventures of Sabrina, Nightflyers, Helix, Manifest, The OA |
| Breaking Bad | Ozark, The Blacklist, House of Cards, Designated Survivor |
| Narcos | Marvel's Luke Cage, Queen of the South, Altered Carbon, Person of Interest |
| Sacred Games | She, Delhi Crime, Jamtara, Babylon Berlin |
| Dick Johnson Is Dead | Giving Voice, The Social Dilemma, Crip Camp |

Full lists: `outputs/sample_recommendations.csv`.

## 5. Evaluation results (K = 10, 1,500 query titles)

| Model | Precision@10 | Strict P@10 (Jaccard >= 0.5) | Mean genre Jaccard | Same-type | Diversity | Coverage |
|---|---|---|---|---|---|---|
| Random baseline | 0.246 | 0.083 | 0.094 | 0.579 | 0.811 | 0.822 |
| Genres only | 1.000 | 0.997 | 0.966 | 1.000 | 0.013 | 0.456 |
| Genres + type | 1.000 | 0.997 | 0.966 | 1.000 | 0.010 | 0.492 |
| **Full weighted model** | **1.000** | **0.996** | **0.960** | **1.000** | **0.099** | **0.741** |
| No genres (cold-start) | 0.689 | 0.405 | 0.358 | 1.000 | 0.140 | 0.614 |

Metric definitions: *Precision@K* = share of recommendations sharing at least one genre with the query; *Strict* requires Jaccard genre overlap >= 0.5; *Diversity* = 1 - mean pairwise similarity inside a list; *Coverage* = share of the catalogue that appears in at least one list.

### How to read this honestly
- Genres are both an input and the thing measured, so the near-perfect precision of the genre models is **expected, not impressive**. It confirms the similarity math works.
- The more informative rows are **Random baseline vs. No genres**: with genres removed, director, country, rating and duration alone still reach 0.689 precision versus 0.246 for random - those attributes carry real signal.
- The **Full weighted model** is the best trade-off: it keeps genre accuracy (1.000) but raises diversity from 0.013 to 0.099 and coverage from 0.456 to 0.741 versus genres-only, so it does not keep recommending the same narrow set of titles.
- Offline proxy metrics cannot replace user testing (click-through, watch time).

## 6. Version 2 upgrades (`netflix_recommender_v2.py`)

Run `python netflix_recommender_v2.py` (about 15 seconds). It is self-contained and reads `Dataset.csv`.

**1. Multi-title recommendations.** `recommend_multi(["Narcos", "Sacred Games", "Ozark"])` averages the feature vectors of the liked titles into one taste profile and ranks everything against it. The terminal also asks you for titles (separate with `;`). The Streamlit `app.py` now lets you pick up to 3 titles.

**2. Hold-out evaluation.** For 1,000 titles with at least two genre tags, one random tag is hidden from the query; the candidates keep all their tags. Metrics: HitRate@10, Precision@10 for the hidden genre, NDCG@10 (relevance = genre Jaccard overlap with the full genre set), Diversity, Coverage.

**3. Charts** (in `outputs/`): `v2_genre_heatmap.png`, `v2_genre_network.png`, `v2_top_countries.png`, `v2_catalogue_map.png`, `v2_method_comparison.png`. (The countries chart is a bar chart, not a map, to avoid extra geo libraries.)

**4. Method comparison** (K=10, 1,000 queries):

| Method | HitRate@10 | Precision@10 (hidden genre) | NDCG@10 | Diversity | Coverage |
|---|---|---|---|---|---|
| Random baseline | 0.628 | 0.143 | 0.105 | n/a | 0.682 |
| Metadata only (no genres) | 0.864 | 0.461 | 0.366 | 0.145 | 0.515 |
| Genres only (TF-IDF) | 0.170 | 0.122 | 0.627 | 0.029 | 0.261 |
| CountVectorizer (binary) | 0.327 | 0.143 | 0.640 | 0.137 | 0.436 |
| TF-IDF + SVD (100 dims) | 0.341 | 0.181 | 0.647 | 0.084 | 0.459 |
| **TF-IDF weighted (final model)** | 0.361 | 0.186 | **0.649** | 0.131 | 0.459 |

### How to read the v2 results honestly
- **NDCG@10 is the main ranking-quality score.** The weighted TF-IDF model is best (0.649), narrowly ahead of TF-IDF + SVD (0.647) and CountVectorizer (0.640). The differences between these three are small, so the fair conclusion is that all three work similarly and weighted TF-IDF is slightly ahead. All are far above random (0.105).
- **The hidden-genre numbers measure something different.** They test whether a model can guess a missing genre tag. "Metadata only" wins there (0.864 hit rate) because it ignores genres completely, so it is not pulled toward titles that match the *remaining* tags. This does not make it a better recommender: its NDCG is only 0.366. It does show that type, country, director, rating and duration carry real signal (precision 0.461 vs. 0.143 for random).
- **The random baseline has a high HitRate (0.628)** because common tags like "International Movies" appear on many titles, so HitRate is easy to get by chance; compare Precision and NDCG instead.
- Limits: the dataset has no user watch history or plot text, so all metrics are offline proxies.

## 7. Limitations and next steps
- Add plot descriptions / cast if available and use TF-IDF on text or sentence embeddings.
- Many titles share identical genre tags, so ties are broken by weak features; richer text would help.
- Add a hybrid layer (collaborative filtering) once user interaction data exists.
- Wrap `recommend()` in a Streamlit or Flask app.

## 8. How to run
```bash
pip install pandas numpy scikit-learn scipy matplotlib
python netflix_recommender.py
python netflix_recommender_v2.py
```
Outputs are written to `outputs/` (`evaluation_results.csv`, `sample_recommendations.csv`, 4 charts). Run the file in a terminal and it will also prompt for any title you type.

## 9. Live demo (Streamlit)
```bash
pip install streamlit
streamlit run app.py
```
Deploy free on Streamlit Community Cloud (push to GitHub, main file `app.py`).

## 10. No-install live demo (GitHub Pages)
`demo/index.html` is a single self-contained page (recommendations precomputed with the same model by `build_demo.py`). Upload it to GitHub, then Settings > Pages > Deploy from branch `main` / root. Live link: `https://YOUR-USERNAME.github.io/netflix-recommender/demo/`

## 11. Files
- `app.py`, `requirements.txt` - Streamlit web app
- `netflix_recommender_v2.py` - upgraded pipeline (multi-title, hold-out evaluation, method comparison, extra charts)
- `netflix_recommender.py` - basic pipeline (cell markers `# %%` work in VS Code / Jupyter)
- `demo/index.html`, `build_demo.py`, `demo_template.html` - static live demo
- `Dataset.csv` - input data
- `outputs/` - results and charts
