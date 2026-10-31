"""Streamlit web app for the Netflix content-based recommender.
Run locally:  streamlit run app.py
"""
import html

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")

WEIGHTS = {"genre_text": 3.0, "type_text": 1.5, "country_text": 1.0,
           "director_text": 1.0, "rating_text": 0.7, "duration_bucket": 0.5, "era": 0.5}


def tokens(text):
    return [t.strip().lower().replace(" ", "_").replace("&", "and").replace("'", "")
            for t in str(text).split(",") if t.strip()]


def comma_tokenizer(s):
    return [t for t in s.split(",") if t]


def duration_bucket(row):
    n = row["duration_num"]
    if pd.isna(n):
        return "dur_unknown"
    if row["type"] == "Movie":
        return "movie_short" if n < 90 else "movie_medium" if n <= 120 else "movie_long"
    return "show_1season" if n == 1 else "show_2to3seasons" if n <= 3 else "show_4plus"


@st.cache_resource
def load():
    df = pd.read_csv("Dataset.csv").drop_duplicates(subset="title").reset_index(drop=True)
    for c in ["director", "country", "rating", "listed_in", "duration"]:
        df[c] = df[c].astype(str).str.strip()
    df[["director", "country"]] = df[["director", "country"]].replace("Not Given", "")
    df["genre_text"] = df["listed_in"].apply(lambda s: ",".join(tokens(s)))
    df["director_text"] = df["director"].apply(lambda s: ",".join(tokens(s)))
    df["country_text"] = df["country"].apply(lambda s: ",".join(tokens(s)))
    df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df["duration_bucket"] = df.apply(duration_bucket, axis=1)
    df["era"] = pd.cut(df["release_year"], [0, 1999, 2009, 2014, 2018, 2100],
                       labels=["pre2000", "2000s", "early2010s", "late2010s", "recent"]).astype(str)
    df["type_text"] = df["type"].str.lower().str.replace(" ", "_")
    df["rating_text"] = df["rating"].str.lower().str.replace("-", "")

    blocks = []
    for col, w in WEIGHTS.items():
        vec = TfidfVectorizer(tokenizer=comma_tokenizer, token_pattern=None, lowercase=False)
        blocks.append(vec.fit_transform(df[col].fillna("")) * w)
    X = hstack(blocks).tocsr()
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1
    return df, X.multiply(1 / norms[:, None]).tocsr()


df, X = load()

CSS = """<style>
#MainMenu, footer, header {visibility:hidden}
.stApp{background:radial-gradient(1100px 520px at 12% -10%,rgba(229,9,20,.38),transparent 60%),
 radial-gradient(900px 500px at 100% 0%,rgba(110,0,90,.28),transparent 55%),#0b0b10}
.stApp, .stApp p, .stApp label, .stApp span{color:#f1f1f4}
.block-container{max-width:980px;padding-top:1.8rem}
.hero{padding:28px 30px;border-radius:20px;margin-bottom:22px;border:1px solid rgba(255,255,255,.09);
 background:linear-gradient(135deg,rgba(229,9,20,.20),rgba(255,255,255,.03))}
.hero h1{margin:0;font-size:2.4rem;font-weight:800;background:linear-gradient(90deg,#ff3b44,#ffc2a8);
 -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{margin:8px 0 0;color:#b9b9c6!important}
.stats{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap}
.stat{padding:8px 16px;border-radius:12px;background:rgba(255,255,255,.07);font-size:.85rem}
.stat b{color:#ff4d55;font-size:1.1rem}
div[data-baseweb="select"]>div{background:#1a1a22!important;border:1px solid #33333f!important;border-radius:12px!important}
div[data-baseweb="tag"]{background:#E50914!important}
.liked{margin:20px 0 10px;font-size:1.2rem;font-weight:700}.liked span{color:#ff4d55}
.card{display:flex;gap:16px;align-items:center;padding:15px 20px;margin-bottom:11px;border-radius:16px;
 background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.07);transition:all .2s}
.card:hover{transform:translateY(-2px);border-color:#E50914;background:rgba(229,9,20,.09);
 box-shadow:0 8px 24px rgba(229,9,20,.18)}
.rank{font-size:1.6rem;font-weight:800;color:#E50914;width:40px;text-align:center}
.info{flex:1}.ttl{font-size:1.08rem;font-weight:700}
.meta{color:#a8a8b6!important;font-size:.82rem;margin-top:3px}
.pill{display:inline-block;padding:2px 11px;margin:7px 6px 0 0;border-radius:999px;font-size:.72rem;
 background:rgba(229,9,20,.18);color:#ff9aa0!important;border:1px solid rgba(229,9,20,.4)}
.match{width:125px;text-align:right}.mbar{height:7px;border-radius:4px;background:rgba(255,255,255,.12);overflow:hidden}
.mbar i{display:block;height:100%;background:linear-gradient(90deg,#E50914,#ff9a5c)}
.mpct{font-size:.8rem;color:#d0d0da;margin-top:5px}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"""<div class="hero"><h1>🎬 Netflix Recommender</h1>
<p>Content-based suggestions powered by TF-IDF features and cosine similarity.</p>
<div class="stats"><div class="stat"><b>{len(df):,}</b> titles</div>
<div class="stat"><b>{X.shape[1]:,}</b> features</div><div class="stat"><b>7</b> content signals</div></div></div>""",
            unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 1, 1])
titles = col1.multiselect("Pick 1-3 titles you liked", sorted(df["title"]), default=["Stranger Things"],
                          max_selections=3, placeholder="Start typing a title...")
n = col2.slider("How many?", 5, 20, 10)
same_type = col3.checkbox("Same type only")


def card(i, rank, score):
    r = df.loc[i]
    pills = "".join(f'<span class="pill">{html.escape(g.strip())}</span>' for g in r["listed_in"].split(","))
    return f"""<div class="card"><div class="rank">{rank}</div><div class="info">
<div class="ttl">{html.escape(r['title'])}</div>
<div class="meta">{html.escape(r['type'])} · {html.escape(r['country'] or 'N/A')} · {r['release_year']} · {html.escape(r['rating'])}</div>
<div>{pills}</div></div><div class="match"><div class="mbar"><i style="width:{max(score, 0) * 100:.0f}%"></i></div>
<div class="mpct">{score * 100:.0f}% match</div></div></div>"""


if titles:
    idxs = [int(df.index[df["title"] == t][0]) for t in titles]
    prof = np.asarray(X[idxs].mean(axis=0)).ravel()      # taste profile = average of liked titles
    prof /= (np.linalg.norm(prof) or 1.0)
    scores = np.asarray(X @ prof).ravel()
    scores[idxs] = -1
    if same_type:
        scores[(df["type"] != df.loc[idxs[0], "type"]).values] = -1
    top = np.argsort(-scores)[:n]
    liked = " + ".join(html.escape(t) for t in titles)
    st.markdown(f'<div class="liked">Because you liked: <span>{liked}</span></div>', unsafe_allow_html=True)
    st.markdown("".join(card(int(i), k + 1, float(scores[i])) for k, i in enumerate(top)), unsafe_allow_html=True)
else:
    st.info("Pick at least one title to get recommendations.")
