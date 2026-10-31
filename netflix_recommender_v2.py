"""
Netflix Content-Based Recommendation System  -  VERSION 2 (upgraded)
====================================================================
Adds to the basic project:
  1. Multi-title recommendations   (taste profile from 2-3 favourite titles)
  2. Hold-out evaluation           (hide one genre tag, test if it is recovered)
                                   + HitRate, NDCG, Precision, Diversity, Coverage
  3. More charts                   (genre heatmap, genre network, countries, SVD map)
  4. Method comparison             (TF-IDF vs CountVectorizer vs TF-IDF+SVD vs others)

Run:  python netflix_recommender_v2.py        (needs Dataset.csv in the same folder)
"""
import difflib
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_PATH, OUT_DIR = "Dataset.csv", "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(42)
RED = "#E50914"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ============================ 1. DATA + FEATURES =============================
def to_tokens(text):
    return [t.strip().lower().replace(" ", "_").replace("&", "and").replace("'", "")
            for t in str(text).split(",") if t.strip()]


def comma_tokenizer(s):
    return [t for t in s.split(",") if t]


df = pd.read_csv(DATA_PATH).drop_duplicates(subset="title").reset_index(drop=True)
for c in ["director", "country", "rating", "listed_in", "duration"]:
    df[c] = df[c].astype(str).str.strip()
df[["director", "country"]] = df[["director", "country"]].replace("Not Given", "")
df["title_clean"] = df["title"].str.strip().str.lower()
df["genre_text"] = df["listed_in"].apply(lambda s: ",".join(to_tokens(s)))
df["director_text"] = df["director"].apply(lambda s: ",".join(to_tokens(s)))
df["country_text"] = df["country"].apply(lambda s: ",".join(to_tokens(s)))
df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)


def duration_bucket(r):
    n = r["duration_num"]
    if pd.isna(n):
        return "dur_unknown"
    if r["type"] == "Movie":
        return "movie_short" if n < 90 else "movie_medium" if n <= 120 else "movie_long"
    return "show_1season" if n == 1 else "show_2to3seasons" if n <= 3 else "show_4plus"


df["duration_bucket"] = df.apply(duration_bucket, axis=1)
df["era"] = pd.cut(df["release_year"], [0, 1999, 2009, 2014, 2018, 2100],
                   labels=["pre2000", "2000s", "early2010s", "late2010s", "recent"]).astype(str)
df["type_text"] = df["type"].str.lower().str.replace(" ", "_")
df["rating_text"] = df["rating"].str.lower().str.replace("-", "")
log(f"Loaded {len(df)} titles")

genre_lists = df["genre_text"].apply(comma_tokenizer).tolist()
genre_sets = [set(g) for g in genre_lists]

WEIGHTS = {"genre_text": 3.0, "type_text": 1.5, "country_text": 1.0, "director_text": 1.0,
           "rating_text": 0.7, "duration_bucket": 0.5, "era": 0.5}


# ============================ 2. FEATURISER ==================================
class Featurizer:
    """Builds the weighted feature matrix. kind = 'tfidf' | 'count'; svd_dims optional."""

    def __init__(self, weights, kind="tfidf", svd_dims=None):
        self.weights, self.vecs, self.blocks = weights, {}, {}
        for col, w in weights.items():
            if kind == "count":
                v = CountVectorizer(tokenizer=comma_tokenizer, token_pattern=None,
                                    lowercase=False, binary=True)
            else:
                v = TfidfVectorizer(tokenizer=comma_tokenizer, token_pattern=None, lowercase=False)
            self.blocks[col] = (v.fit_transform(df[col].fillna("")) * w).tocsr()
            self.vecs[col] = v
        X = normalize(hstack(list(self.blocks.values())).tocsr())
        self.svd = TruncatedSVD(svd_dims, random_state=42) if svd_dims else None
        self.M = normalize(self.svd.fit_transform(X)) if self.svd else X
        self.X = X

    def query_vec(self, i, genre_text=None):
        """Vector of title i. If genre_text is given it replaces the title's genres."""
        parts = []
        for col, w in self.weights.items():
            if col == "genre_text" and genre_text is not None:
                parts.append(self.vecs[col].transform([genre_text]) * w)
            else:
                parts.append(self.blocks[col][i])
        q = normalize(hstack(parts).tocsr())
        return normalize(self.svd.transform(q)) if self.svd else q

    def scores(self, q):
        s = self.M @ q.T
        return (s.toarray() if hasattr(s, "toarray") else np.asarray(s)).ravel()


# ============================ 3. RECOMMENDERS ================================
main = Featurizer(WEIGHTS)                      # the project's main model
title_to_idx = pd.Series(df.index.values, index=df["title_clean"])


def find_title(query):
    q = query.strip().lower()
    if q in title_to_idx.index:
        return int(title_to_idx[q])
    close = difflib.get_close_matches(q, title_to_idx.index, n=1, cutoff=0.5)
    if not close:
        raise ValueError(f"No title similar to '{query}' found.")
    print(f"  (using closest match: '{df.loc[title_to_idx[close[0]], 'title']}')")
    return int(title_to_idx[close[0]])


def recommend_multi(titles, n=10, same_type_only=False, feat=main):
    """Taste profile = average vector of the liked titles; works for 1, 2, 3+ titles."""
    idxs = [find_title(t) for t in titles]
    prof = np.asarray(feat.M[idxs].mean(axis=0)).ravel()
    prof /= (np.linalg.norm(prof) or 1.0)
    s = np.asarray(feat.M @ prof).ravel()
    s[idxs] = -1
    if same_type_only:
        s[(df["type"] != df.loc[idxs[0], "type"]).values] = -1
    top = np.argsort(-s)[:n]
    out = df.loc[top, ["title", "type", "listed_in", "country", "release_year", "rating"]].copy()
    out.insert(1, "similarity", s[top].round(3))
    liked = " + ".join(df.loc[i, "title"] for i in idxs)
    return liked, out.reset_index(drop=True)


def show_multi(titles, n=8):
    liked, out = recommend_multi(titles, n)
    print("\n" + "=" * 92 + f"\nBecause you liked: {liked}\n" + "=" * 92)
    print(out.to_string(index=False))
    return liked, out


MULTI_DEMOS = [["Stranger Things", "Breaking Bad"],
               ["Narcos", "Sacred Games", "Ozark"],
               ["The Twilight Saga: Breaking Dawn: Part 1", "Dear John"],
               ["Dick Johnson Is Dead", "The Social Dilemma"]]
frames = []
for combo in MULTI_DEMOS:
    try:
        liked, out = show_multi(combo)
        frames.append(out.assign(liked=liked))
    except ValueError as e:
        print(e)
pd.concat(frames).to_csv(f"{OUT_DIR}/sample_multi_recommendations.csv", index=False)

# ============================ 4. EVALUATION ==================================
# HOLD-OUT TEST: for a title with >= 2 genre tags we HIDE one random tag from the
# query (candidates keep all their tags). A good model should still recommend titles
# that carry the hidden tag - this removes the "genres measured by genres" problem.
#   HitRate@K   at least one of the K recs has the hidden tag
#   Precision@K share of the K recs that have the hidden tag
#   NDCG@K      ranking quality, relevance = Jaccard overlap with the FULL genre set
#   Diversity   1 - mean pairwise similarity inside a rec list
#   Coverage    share of catalogue that appears in any rec list
K, N_QUERIES = 10, 1000
G_vec = CountVectorizer(tokenizer=comma_tokenizer, token_pattern=None, lowercase=False, binary=True)
G = G_vec.fit_transform(df["genre_text"]).tocsr().astype(float)
G_size = np.asarray(G.sum(axis=1)).ravel()
multi = [i for i, g in enumerate(genre_lists) if len(g) >= 2]
q_ids = RNG.choice(multi, size=N_QUERIES, replace=False)
hidden = {int(i): RNG.choice(genre_lists[i]) for i in q_ids}
disc = 1 / np.log2(np.arange(2, K + 2))


def jaccard_all(i):
    inter = (G @ G[i].T).toarray().ravel()
    return inter / (G_size + G_size[i] - inter)


def evaluate(feat, name):
    hit, prec, ndcg, div, cov = [], [], [], [], set()
    for i in q_ids:
        i = int(i)
        h = hidden[i]
        rest = ",".join(g for g in genre_lists[i] if g != h)
        if feat is None:                                    # random baseline
            top = RNG.choice(np.delete(np.arange(len(df)), i), size=K, replace=False)
            sub = None
        else:
            s = feat.scores(feat.query_vec(i, rest if "genre_text" in feat.weights else None))
            s[i] = -1
            top = np.argsort(-s)[:K]
            sub = feat.M[top]
        has = np.array([h in genre_sets[r] for r in top], dtype=float)
        hit.append(has.max()); prec.append(has.mean()); cov.update(top.tolist())
        rel_all = jaccard_all(i); rel_all[i] = -1
        ideal = np.sort(rel_all)[::-1][:K]
        ndcg.append((rel_all[top] * disc).sum() / (ideal * disc).sum())
        if sub is not None:
            sim = sub @ sub.T
            sim = sim.toarray() if hasattr(sim, "toarray") else np.asarray(sim)
            div.append(1 - (sim.sum() - K) / (K * (K - 1)))
    return {"method": name, "HitRate@10": np.mean(hit), "Precision@10 (hidden genre)": np.mean(prec),
            "NDCG@10": np.mean(ndcg), "Diversity": np.mean(div) if div else np.nan,
            "Coverage": len(cov) / len(df)}


# ============================ 5. METHOD COMPARISON ===========================
log("Building comparison models (TF-IDF, Count, SVD, ...)")
METHODS = {
    "Random baseline": None,
    "Metadata only (no genres)": Featurizer({k: v for k, v in WEIGHTS.items() if k != "genre_text"}),
    "Genres only (TF-IDF)": Featurizer({"genre_text": 1.0}),
    "CountVectorizer (binary)": Featurizer(WEIGHTS, kind="count"),
    "TF-IDF + SVD (100 dims)": Featurizer(WEIGHTS, svd_dims=100),
    "TF-IDF weighted (ours)": main,
}
rows = []
for name, feat in METHODS.items():
    log(f"Evaluating: {name}")
    rows.append(evaluate(feat, name))
results = pd.DataFrame(rows).set_index("method")
print("\nHOLD-OUT EVALUATION (K=10, %d queries, one genre tag hidden per query)\n" % N_QUERIES)
print(results.round(3).to_string())
results.round(4).to_csv(f"{OUT_DIR}/evaluation_v2_results.csv")

# ============================ 6. CHARTS ======================================
plt.style.use("seaborn-v0_8-whitegrid")
log("Drawing charts")

# 6a. Method comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
results[["HitRate@10", "Precision@10 (hidden genre)", "NDCG@10"]].plot.bar(
    ax=ax[0], color=[RED, "#221f1f", "#b3b3b3"], width=0.8)
ax[0].set_title("Accuracy on the hold-out test (higher = better)"); ax[0].set_xlabel("")
ax[0].set_ylim(0, 1.05); ax[0].tick_params(axis="x", rotation=30)
plt.setp(ax[0].get_xticklabels(), ha="right")
results[["Diversity", "Coverage"]].plot.bar(ax=ax[1], color=["#221f1f", RED], width=0.7)
ax[1].set_title("Variety of recommendations"); ax[1].set_xlabel("")
ax[1].tick_params(axis="x", rotation=30); plt.setp(ax[1].get_xticklabels(), ha="right")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/v2_method_comparison.png", dpi=150); plt.close()

# 6b. Genre co-occurrence heatmap (top 15 genres)
names = np.array(G_vec.get_feature_names_out())
freq = np.asarray(G.sum(axis=0)).ravel()
top_g = np.argsort(-freq)[:15]
co = (G.T @ G).toarray()[np.ix_(top_g, top_g)]
pretty = [n.replace("_", " ").title() for n in names[top_g]]
cos = co / np.sqrt(np.outer(np.diag(co), np.diag(co)))          # normalised, 1 on the diagonal
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cos, cmap="Reds"); plt.colorbar(im, fraction=0.04, label="Co-occurrence strength")
ax.set_xticks(range(15)); ax.set_xticklabels(pretty, rotation=60, ha="right")
ax.set_yticks(range(15)); ax.set_yticklabels(pretty); ax.grid(False)
ax.set_title("Which genres appear together? (top 15 genres)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/v2_genre_heatmap.png", dpi=150); plt.close()

# 6c. Genre network (top 20 genres, strongest links)
top20 = np.argsort(-freq)[:20]
co20 = (G.T @ G).toarray()[np.ix_(top20, top20)].astype(float)
ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
pos = np.c_[np.cos(ang), np.sin(ang)]
pairs = sorted(((co20[a, b], a, b) for a in range(20) for b in range(a + 1, 20)), reverse=True)[:45]
mx = pairs[0][0]
fig, ax = plt.subplots(figsize=(10, 10))
for w, a, b in pairs:
    ax.plot(*zip(pos[a], pos[b]), color=RED, alpha=0.15 + 0.6 * w / mx, lw=0.5 + 6 * w / mx, zorder=1)
ax.scatter(pos[:, 0], pos[:, 1], s=freq[top20] / 3, color="#221f1f", zorder=2)
for k in range(20):
    ha = "left" if pos[k, 0] > 0.3 else "right" if pos[k, 0] < -0.3 else "center"
    ax.text(pos[k, 0] * 1.12, pos[k, 1] * 1.12, names[top20[k]].replace("_", " ").title(),
            ha=ha, va="center", fontsize=9)
ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.45, 1.45); ax.axis("off")
ax.set_title("Genre network: thicker line = genres tagged together more often")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/v2_genre_network.png", dpi=150); plt.close()

# 6d. Top countries (movies vs shows)
cc = df.assign(c=df["country"].str.split(",")).explode("c")
cc["c"] = cc["c"].str.strip(); cc = cc[cc["c"] != ""]
tab = cc.groupby(["c", "type"]).size().unstack(fill_value=0)
tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).head(10).index][::-1]
fig, ax = plt.subplots(figsize=(9, 5.5))
tab.plot.barh(stacked=True, ax=ax, color=[RED, "#221f1f"])
ax.set_title("Top 10 content-producing countries"); ax.set_ylabel(""); ax.set_xlabel("Titles")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/v2_top_countries.png", dpi=150); plt.close()

# 6e. 2-D map of the catalogue (SVD), coloured by main genre
xy = TruncatedSVD(2, random_state=42).fit_transform(main.X)
first = pd.Series([g[0] if g else "other" for g in genre_lists])
keep = first.value_counts().head(7).index
lab = first.where(first.isin(keep), "other")
sel = RNG.choice(len(df), size=5000, replace=False)
fig, ax = plt.subplots(figsize=(9, 7))
palette = [RED, "#221f1f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#cccccc"]
for c, col in zip(list(keep) + ["other"], palette):
    m = sel[(lab.values[sel] == c)]
    ax.scatter(xy[m, 0], xy[m, 1], s=7, alpha=0.6, color=col, label=c.replace("_", " ").title())
ax.legend(markerscale=3, fontsize=8); ax.set_title("Catalogue map (2-D SVD of the feature matrix)")
ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/v2_catalogue_map.png", dpi=150); plt.close()

log(f"Done. Charts + CSVs saved in ./{OUT_DIR}/")

# ============================ 7. INTERACTIVE =================================
if __name__ == "__main__":
    import sys
    if sys.stdin.isatty():
        print("\nTry it yourself. Enter 1-3 titles separated by a semicolon (;)")
        while True:
            line = input("\nTitles (or press Enter to quit): ").strip()
            if not line:
                break
            try:
                show_multi([t for t in line.split(";") if t.strip()], n=10)
            except ValueError as e:
                print(e)
