import streamlit as st
import pandas as pd
import feedparser
from datetime import datetime, timezone
from urllib.parse import quote
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="NVIDIA News Dashboard", layout="wide")

# -------------------------
# Helpers
# -------------------------
def google_news_rss_url(query: str, lang="en", country="US") -> str:
    # Google News RSS search endpoint
    q = quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={country}&ceid={country}:{lang}"

def safe_get(entry, key, default=None):
    return entry.get(key, default) if isinstance(entry, dict) else default

def parse_feed(url: str, source_name: str) -> pd.DataFrame:
    feed = feedparser.parse(url)

    rows = []
    for e in feed.entries:
        title = safe_get(e, "title", "").strip()
        link = safe_get(e, "link", "").strip()

        # published_parsed is usually a struct_time
        published_dt = None
        if safe_get(e, "published_parsed"):
            published_dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        elif safe_get(e, "updated_parsed"):
            published_dt = datetime(*e.updated_parsed[:6], tzinfo=timezone.utc)

        summary = (safe_get(e, "summary", "") or "").strip()

        rows.append({
            "source": source_name,
            "title": title,
            "link": link,
            "published_utc": published_dt,
            "summary": summary
        })

    df = pd.DataFrame(rows)
    return df

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # sentiment polarity: -1 (negative) to +1 (positive)
    def score(text):
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            return None

    df = df.copy()
    df["sentiment"] = df["title"].apply(score)
    return df

def dedupe_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_norm"] = df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["title_norm", "link"]).drop(columns=["title_norm"])
    return df

# -------------------------
# Data Loading
# -------------------------
@st.cache_data(ttl=900)  # cache for 15 minutes
def load_news(query: str, lang="en", country="US") -> pd.DataFrame:
    feeds = [
        ("Google News", google_news_rss_url(query, lang=lang, country=country)),
        # Add more RSS sources here if you want:
        # ("Some Tech RSS", "https://example.com/rss"),
    ]

    all_dfs = []
    errors = []

    for name, url in feeds:
        try:
            df = parse_feed(url, name)
            all_dfs.append(df)
        except Exception as ex:
            errors.append((name, str(ex)))

    if not all_dfs:
        return pd.DataFrame(), errors

    df = pd.concat(all_dfs, ignore_index=True)
    df = dedupe_news(df)

    # Convert to local-friendly display (still keep UTC column)
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
    df["published"] = df["published_utc"].dt.tz_convert("Asia/Karachi")
    df = df.sort_values("published_utc", ascending=False)

    df = add_sentiment(df)

    return df, errors

# -------------------------
# UI
# -------------------------
st.title("🟩 NVIDIA News Scraper + Dashboard (RSS based)")
st.caption("Pulls headlines via RSS (fast + reliable). Use filters, sentiment, and word cloud.")

with st.sidebar:
    st.header("Settings")
    query = st.text_input("Search query", value="NVIDIA OR NVDA")
    lang = st.selectbox("Language", ["en"], index=0)
    country = st.selectbox("Country edition", ["US", "GB", "IN", "PK"], index=0)
    refresh = st.button("Refresh now")

if refresh:
    st.cache_data.clear()

df, errors = load_news(query=query, lang=lang, country=country)

if errors:
    with st.expander("Feed errors (non-fatal)"):
        for name, msg in errors:
            st.write(f"**{name}**: {msg}")

if df.empty:
    st.warning("No news loaded. Try a different query or country edition.")
    st.stop()

# Filters
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    keyword = st.text_input("Filter keyword", value="")
with col2:
    sources = st.multiselect("Sources", sorted(df["source"].unique().tolist()),
                             default=sorted(df["source"].unique().tolist()))
with col3:
    sort = st.selectbox("Sort", ["Newest", "Oldest"], index=0)
with col4:
    min_sent = st.slider("Min sentiment", -1.0, 1.0, -1.0, 0.05)

f = df[df["source"].isin(sources)].copy()

if keyword.strip():
    k = keyword.strip().lower()
    f = f[f["title"].str.lower().str.contains(k, na=False) | f["summary"].str.lower().str.contains(k, na=False)]

f = f[f["sentiment"].fillna(0) >= min_sent]

if sort == "Oldest":
    f = f.sort_values("published_utc", ascending=True)
else:
    f = f.sort_values("published_utc", ascending=False)

# KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Total headlines", len(f))
k2.metric("Sources", f["source"].nunique())
k3.metric("Avg sentiment", round(float(f["sentiment"].dropna().mean()) if f["sentiment"].notna().any() else 0, 3))

# Layout
left, right = st.columns([2.2, 1])

with left:
    st.subheader("Headlines")
    # Display clickable rows
    for _, row in f.head(50).iterrows():
        published_str = row["published"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["published"]) else "N/A"
        st.markdown(
            f"""
            **[{row['title']}]({row['link']})**  
            {published_str} • *{row['source']}* • sentiment: `{row['sentiment']:.2f}`  
            """.strip()
        )
        st.divider()

    st.download_button(
        "Download CSV",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="nvidia_news.csv",
        mime="text/csv"
    )

with right:
    st.subheader("Word Cloud (titles)")
    text = " ".join(f["title"].dropna().astype(str).tolist())
    if text.strip():
        wc = WordCloud(width=800, height=500, background_color="white").generate(text)
        fig = plt.figure()
        plt.imshow(wc)
        plt.axis("off")
        st.pyplot(fig)
    else:
        st.info("Not enough text for word cloud.")

    st.subheader("Sentiment distribution")
    fig2 = plt.figure()
    f["sentiment"].dropna().plot(kind="hist", bins=20)
    st.pyplot(fig2)
