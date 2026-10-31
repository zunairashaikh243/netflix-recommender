"""
Netflix Content-Based Recommendation System
===========================================
Internship Task 1 (Easy)

Workflow
  Step 1  Prepare content-related features
  Step 2  Convert text data into machine-readable format (TF-IDF)
  Step 3  Calculate content similarity scores (cosine similarity)
  Step 4  Generate recommendations for selected titles
  Step 5  Evaluate recommendation quality

Run:  python netflix_recommender.py
(The "# %%" markers also let you run it cell-by-cell in VS Code / Jupyter.)
"""

# %% Imports ------------------------------------------------------------------
import difflib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "Dataset.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(42)

# %% STEP 1 - Prepare content-related features --------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
print(df.columns.tolist())

# 1a. Cleaning ---------------------------------------------------------------
df = df.drop_duplicates(subset="title").reset_index(drop=True)   # 3 duplicate titles
for col in ["director", "country", "rating", "listed_in", "duration"]:
    df[col] = df[col].astype(str).str.strip()
# The dataset marks missing values with the literal string "Not Given"
df[["director", "country"]] = df[["director", "country"]].replace("Not Given", "")
df["title_clean"] = df["title"].str.strip().str.lower()

# 1b. Feature engineering ----------------------------------------------------
# Genres: "Dramas, International Movies" -> ["dramas", "international_movies"]
def to_tokens(text: str) -> list[str]:
    return [t.strip().lower().replace(" ", "_").replace("&", "and").replace("'", "")
            for t in text.split(",") if t.strip()]

df["genre_text"] = df["listed_in"].apply(lambda s: ",".join(to_tokens(s)))
df["director_text"] = df["director"].apply(lambda s: ",".join(to_tokens(s)))
df["country_text"] = df["country"].apply(lambda s: ",".join(to_tokens(s)))

# Duration: movies -> minutes, TV shows -> seasons. Turn both into buckets.
df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)

def duration_bucket(row) -> str:
    n = row["duration_num"]
    if pd.isna(n):
        return "dur_unknown"
    if row["type"] == "Movie":
        return "movie_short" if n < 90 else "movie_medium" if n <= 120 else "movie_long"
    return "show_1season" if n == 1 else "show_2to3seasons" if n <= 3 else "show_4plus"

df["duration_bucket"] = df.apply(duration_bucket, axis=1)

# Release era bucket (a 1995 film and a 2021 film are rarely "similar")
df["era"] = pd.cut(df["release_year"], [0, 1999, 2009, 2014, 2018, 2100],
                   labels=["pre2000", "2000s", "early2010s", "late2010s", "recent"]).astype(str)

df["type_text"] = df["type"].str.lower().str.replace(" ", "_")
df["rating_text"] = df["rating"].str.lower().str.replace("-", "")

print(df[["title", "genre_text", "country_text", "duration_bucket", "era"]].head())

# %% STEP 2 - Convert text to machine-readable format (TF-IDF) ----------------
# One TF-IDF block per feature group. Each block is multiplied by a weight so that
# genres dominate while the other attributes act as tie-breakers.
def comma_tokenizer(s: str) -> list[str]:
    return [t for t in s.split(",") if t]

FEATURE_BLOCKS = {
    #  column          weight
    "genre_text":      3.0,
    "type_text":       1.5,
    "country_text":    1.0,
    "director_text":   1.0,
    "rating_text":     0.7,
    "duration_bucket": 0.5,
    "era":             0.5,
}

def build_matrix(weights: dict):
    """Return an L2-normalised sparse feature matrix built from the weighted blocks."""
    blocks = []
    for col, w in weights.items():
        if w == 0:
            continue
        vec = TfidfVectorizer(tokenizer=comma_tokenizer, token_pattern=None,
                              lowercase=False, norm="l2")
        blocks.append(vec.fit_transform(df[col].fillna("")) * w)
    X = hstack(blocks).tocsr()
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1
    return X.multiply(1 / norms[:, None]).tocsr()          # unit-length rows

X = build_matrix(FEATURE_BLOCKS)
print(f"Feature matrix: {X.shape[0]} titles x {X.shape[1]} features "
      f"(sparsity {100 * (1 - X.nnz / (X.shape[0] * X.shape[1])):.1f}%)")

# %% STEP 3 - Calculate content similarity scores -----------------------------
# Rows are unit-length, so cosine similarity == dot product.
# We compute similarity on demand (one row at a time) instead of storing the full
# 8,790 x 8,790 matrix, which keeps memory low.
def similarity_to_all(idx: int, matrix=None) -> np.ndarray:
    M = X if matrix is None else matrix
    return (M @ M[idx].T).toarray().ravel()

title_to_idx = pd.Series(df.index.values, index=df["title_clean"])

# %% STEP 4 - Generate recommendations ----------------------------------------
def find_title(query: str) -> int:
    """Exact (case-insensitive) match, else closest fuzzy match."""
    q = query.strip().lower()
    if q in title_to_idx.index:
        return int(title_to_idx[q])
    close = difflib.get_close_matches(q, title_to_idx.index, n=1, cutoff=0.5)
    if not close:
        raise ValueError(f"No title similar to '{query}' found.")
    print(f"(No exact match - using closest title: '{df.loc[title_to_idx[close[0]], 'title']}')")
    return int(title_to_idx[close[0]])

def recommend(title: str, n: int = 10, same_type_only: bool = False,
              matrix=None) -> pd.DataFrame:
    idx = find_title(title)
    scores = similarity_to_all(idx, matrix)
    scores[idx] = -1                                   # never recommend the title itself
    if same_type_only:
        scores[(df["type"] != df.loc[idx, "type"]).values] = -1
    top = np.argsort(-scores)[:n]
    out = df.loc[top, ["title", "type", "listed_in", "country", "release_year", "rating"]].copy()
    out.insert(1, "similarity", scores[top].round(3))
    return out.reset_index(drop=True)

def show(title: str, n: int = 8, **kw):
    idx = find_title(title)
    print("\n" + "=" * 90)
    print(f"Because you watched: {df.loc[idx, 'title']}  [{df.loc[idx, 'type']}]  "
          f"-> {df.loc[idx, 'listed_in']}")
    print("=" * 90)
    print(recommend(title, n, **kw).to_string(index=False))

DEMO_TITLES = ["Stranger Things", "Breaking Bad", "The Twilight Saga: Breaking Dawn: Part 1",
               "Narcos", "Sacred Games", "Dick Johnson Is Dead"]
demo_results = {}
for t in DEMO_TITLES:
    try:
        show(t)
        demo_results[t] = recommend(t, 10)
    except ValueError as e:
        print(e)

# Save recommendations for the report
pd.concat({k: v for k, v in demo_results.items()}, names=["query", "rank"]) \
  .reset_index(level=0).to_csv(f"{OUT_DIR}/sample_recommendations.csv", index=False)

# %% STEP 5 - Evaluate recommendation quality ---------------------------------
# The dataset has no user ratings / watch history, so we use OFFLINE PROXY METRICS
# measured over a random sample of query titles (K = 10):
#   Precision@K      share of recs that share >= 1 genre with the query title
#   Strict P@K       share of recs with genre Jaccard overlap >= 0.5
#   Mean Jaccard     average genre overlap between query and recs
#   Same-type rate   recs that are the same Movie / TV Show type
#   Diversity        1 - mean pairwise similarity inside each rec list
#   Coverage         % of the catalogue that appears in at least one rec list
# A RANDOM baseline shows how much better the model is than chance.
K = 10
N_QUERIES = 1500
genre_sets = df["listed_in"].apply(lambda s: set(to_tokens(s))).tolist()
query_ids = RNG.choice(len(df), size=N_QUERIES, replace=False)

def evaluate(matrix, name: str, random_baseline: bool = False) -> dict:
    p_at_k, strict, jac, same_type, div = [], [], [], [], []
    covered = set()
    for q in query_ids:
        if random_baseline:
            recs = RNG.choice(np.delete(np.arange(len(df)), q), size=K, replace=False)
        else:
            s = similarity_to_all(q, matrix)
            s[q] = -1
            recs = np.argsort(-s)[:K]
        covered.update(recs.tolist())
        overlaps = [len(genre_sets[q] & genre_sets[r]) / len(genre_sets[q] | genre_sets[r])
                    for r in recs]
        p_at_k.append(np.mean([o > 0 for o in overlaps]))
        strict.append(np.mean([o >= 0.5 for o in overlaps]))
        jac.append(np.mean(overlaps))
        same_type.append(np.mean(df["type"].values[recs] == df["type"].values[q]))
        sub = matrix[recs]
        sim = (sub @ sub.T).toarray()
        div.append(1 - (sim.sum() - K) / (K * (K - 1)))        # drop the diagonal
    return {"model": name, "Precision@10": np.mean(p_at_k), "Strict P@10 (J>=0.5)": np.mean(strict),
            "Mean genre Jaccard": np.mean(jac), "Same-type rate": np.mean(same_type),
            "Diversity": np.mean(div), "Coverage": len(covered) / len(df)}

# Ablation: which features matter?
VARIANTS = {
    "Genres only":               {"genre_text": 1},
    "Genres + type":             {"genre_text": 3, "type_text": 1.5},
    "Full weighted model":       FEATURE_BLOCKS,
    "No genres (cold-start)":    {k: v for k, v in FEATURE_BLOCKS.items() if k != "genre_text"},
}
rows = [evaluate(X, "Random baseline", random_baseline=True)]
for name, w in VARIANTS.items():
    rows.append(evaluate(build_matrix(w), name))
results = pd.DataFrame(rows).set_index("model")
print("\nEvaluation (K=10, %d random query titles)\n" % N_QUERIES)
print(results.round(3).to_string())
results.round(4).to_csv(f"{OUT_DIR}/evaluation_results.csv")

# %% Visualisations -----------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
RED = "#E50914"

# (a) Top genres
genre_counts = pd.Series([g for s in genre_sets for g in s]).value_counts().head(12)[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(genre_counts.index.str.replace("_", " ").str.title(), genre_counts.values, color=RED)
ax.set_title("Top 12 Genres / Categories in the Catalogue"); ax.set_xlabel("Number of titles")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/top_genres.png", dpi=150); plt.close()

# (b) Movies vs TV shows by release year
fig, ax = plt.subplots(figsize=(8, 4.5))
yearly = df[df.release_year >= 2000].groupby(["release_year", "type"]).size().unstack(fill_value=0)
yearly.plot(ax=ax, color=[RED, "#221f1f"], lw=2)
ax.set_title("Titles by Release Year (2000+)"); ax.set_ylabel("Titles"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/titles_by_year.png", dpi=150); plt.close()

# (c) Model comparison
fig, ax = plt.subplots(figsize=(9, 5))
plot_df = results[["Precision@10", "Strict P@10 (J>=0.5)", "Same-type rate"]]
plot_df.plot.bar(ax=ax, color=[RED, "#221f1f", "#b3b3b3"], width=0.75)
ax.set_title("Recommendation Quality: Baseline vs Model Variants (K=10)")
ax.set_ylabel("Score"); ax.set_xlabel(""); ax.set_ylim(0, 1.05)
plt.xticks(rotation=20, ha="right"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150); plt.close()

# (d) Similarity score distribution for one example title
idx = find_title("Stranger Things")
s = similarity_to_all(idx); s[idx] = np.nan
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(s[~np.isnan(s)], bins=50, color=RED)
ax.set_title("Similarity of Every Title to 'Stranger Things'")
ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Number of titles"); ax.set_yscale("log")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/similarity_distribution.png", dpi=150); plt.close()

print(f"\nSaved charts and CSVs to ./{OUT_DIR}/")

# %% Interactive helper ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    if sys.stdin.isatty():
        while True:
            q = input("\nEnter a Netflix title (or press Enter to quit): ").strip()
            if not q:
                break
            try:
                show(q, n=10)
            except ValueError as e:
                print(e)
