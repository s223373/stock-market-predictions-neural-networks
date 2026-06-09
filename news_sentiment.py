"""
news_sentiment.py
=================
FinBERT-based news sentiment pipeline for stock tickers.

Pulls recent headlines via the Finnhub API, scores them with FinBERT
(a BERT model fine-tuned on financial text), and produces per-bar
sentiment features aligned to a yfinance OHLCV dataframe.

Setup
-----
1. Get a free Finnhub API key at https://finnhub.io  (free tier: 60 req/min)
2. Set it as an environment variable:
       export FINNHUB_API_KEY="your_key_here"
   or pass it directly: SentimentPipeline(ticker, api_key="your_key_here")

3. Install dependencies (first run only):
       pip install transformers torch finnhub-python

Usage in feature_engineering.py
---------------------------------
    from news_sentiment import SentimentPipeline

    pipe = SentimentPipeline("AAPL")
    df   = pipe.add_sentiment_features(df)
    # df now has: news_sentiment_score, news_bull, news_bear, news_article_count

Usage as standalone script
---------------------------
    python news_sentiment.py --ticker AAPL --days 7
"""

import os
import time
import argparse
import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import finnhub
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

FINBERT_MODEL   = "ProsusAI/finbert"   # HuggingFace model ID
BULL_THRESHOLD  =  0.15   # score above this → news_bull = 1
BEAR_THRESHOLD  = -0.15   # score below this → news_bear = 1
MAX_HEADLINES   = 200     # cap per fetch to avoid rate limits
LABEL_MAP       = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# FINBERT SCORER
# ─────────────────────────────────────────────────────────────────────────────

class FinBERTScorer:
    """
    Wraps the ProsusAI/finbert model for batch headline scoring.

    Why FinBERT over general VADER/TextBlob?
    ----------------------------------------
    General sentiment models are trained on movie reviews, social media, etc.
    Financial text has domain-specific language ('missed estimates', 'beat
    expectations', 'raised guidance') that maps poorly to general positive/
    negative sentiment. FinBERT is fine-tuned on 10-K filings, earnings calls,
    and financial news — it handles these phrases correctly.

    Output
    ------
    Per-headline score in [-1, +1]:
        +1 = strongly bullish
         0 = neutral
        -1 = strongly bearish

    The score is the signed softmax probability:
        score = P(positive) - P(negative)
    This preserves confidence — a 90% positive prediction scores +0.9,
    while a 51% positive prediction scores only +0.02.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FinBERT] Loading {FINBERT_MODEL} on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self.model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        self.model.to(self.device)
        self.model.eval()
        # Label order from the model config: positive=0, negative=1, neutral=2
        # Verify this matches ProsusAI/finbert's id2label
        id2label = self.model.config.id2label
        self.pos_idx = [k for k, v in id2label.items() if v.lower() == "positive"][0]
        self.neg_idx = [k for k, v in id2label.items() if v.lower() == "negative"][0]
        print(f"[FinBERT] Ready. pos_idx={self.pos_idx}, neg_idx={self.neg_idx}")

    @torch.no_grad()
    def score_batch(self, headlines: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Score a list of headlines. Returns array of shape (N,) with values in [-1, 1].
        """
        if not headlines:
            return np.array([], dtype=float)

        all_scores = []
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            enc   = self.tokenizer(
                batch,
                padding      = True,
                truncation   = True,
                max_length   = 128,   # headlines are short; 128 is plenty
                return_tensors = "pt",
            ).to(self.device)

            logits = self.model(**enc).logits           # (batch, 3)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

            # score = P(positive) - P(negative) ∈ (-1, 1)
            scores = probs[:, self.pos_idx] - probs[:, self.neg_idx]
            all_scores.append(scores)

        return np.concatenate(all_scores)


# ─────────────────────────────────────────────────────────────────────────────
# FINNHUB NEWS FETCHER
# ─────────────────────────────────────────────────────────────────────────────

class NewsFetcher:
    """
    Fetches company-specific headlines from Finnhub.

    Returns a DataFrame with columns:
        datetime    timezone-aware UTC timestamp of the article
        headline    article title string
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        if not key:
            raise ValueError(
                "Finnhub API key required. Set FINNHUB_API_KEY env var or pass api_key=."
            )
        self.client = finnhub.Client(api_key=key)

    def fetch(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch up to MAX_HEADLINES articles for `ticker` over the last `days_back` days.
        """
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=days_back)

        print(f"[Finnhub] Fetching news for {ticker} from {start} to {end} …")
        try:
            raw = self.client.company_news(
                ticker,
                _from = start.strftime("%Y-%m-%d"),
                to    = end.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            print(f"[Finnhub] API error: {e}")
            return pd.DataFrame(columns=["datetime", "headline"])

        if not raw:
            print(f"[Finnhub] No articles returned for {ticker}.")
            return pd.DataFrame(columns=["datetime", "headline"])

        records = []
        for art in raw[:MAX_HEADLINES]:
            headline = art.get("headline", "").strip()
            ts       = art.get("datetime", 0)
            if headline and ts:
                records.append({
                    "datetime": pd.Timestamp(ts, unit="s", tz="UTC"),
                    "headline": headline,
                })

        df = pd.DataFrame(records)
        print(f"[Finnhub] Retrieved {len(df)} articles.")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class SentimentPipeline:
    """
    End-to-end pipeline: fetch headlines → score → align to OHLCV bars.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g. "AAPL").
    api_key : str, optional
        Finnhub API key. Falls back to FINNHUB_API_KEY env var.
    days_back : int
        How many calendar days of news to fetch. Should match or exceed
        the period used to download OHLCV data (e.g. 30 for "30d").
    device : str, optional
        "cpu" or "cuda". Auto-detected if not specified.
    bull_threshold : float
        Aggregated score above this → news_bull = 1. Default 0.15.
    bear_threshold : float
        Aggregated score below this → news_bear = 1. Default -0.15.
    """

    def __init__(
        self,
        ticker:         str,
        api_key:        Optional[str] = None,
        days_back:      int   = 35,
        device:         Optional[str] = None,
        bull_threshold: float = BULL_THRESHOLD,
        bear_threshold: float = BEAR_THRESHOLD,
    ):
        self.ticker         = ticker
        self.days_back      = days_back
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

        self.fetcher = NewsFetcher(api_key=api_key)
        self.scorer  = FinBERTScorer(device=device)
        self._scored_news: Optional[pd.DataFrame] = None   # cached after first run

    # ── Core method ──────────────────────────────────────────────────────────

    def fetch_and_score(self) -> pd.DataFrame:
        """
        Download and score headlines. Caches result so it only runs once
        per SentimentPipeline instance.

        Returns
        -------
        pd.DataFrame with columns [datetime, headline, sentiment_score]
            sorted by datetime ascending.
        """
        if self._scored_news is not None:
            return self._scored_news

        news = self.fetcher.fetch(self.ticker, days_back=self.days_back)

        if news.empty:
            print("[sentiment] No news found — sentiment features will be 0.")
            self._scored_news = news
            return news

        headlines = news["headline"].tolist()
        scores    = self.scorer.score_batch(headlines)
        news      = news.copy()
        news["sentiment_score"] = scores
        news = news.sort_values("datetime").reset_index(drop=True)

        print(f"[sentiment] Scored {len(news)} headlines.")
        print(f"            Mean score : {scores.mean():.4f}")
        print(f"            Bull count : {(scores >  self.bull_threshold).sum()}")
        print(f"            Bear count : {(scores < self.bear_threshold).sum()}")

        self._scored_news = news
        return news

    # ── Feature generation ───────────────────────────────────────────────────

    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align scored headlines to OHLCV bars and add sentiment features.

        For each bar, we aggregate all headlines published BEFORE that bar
        (lookahead-free) within a rolling window, producing a sentiment signal
        that reflects what the market could have known at that point in time.

        Aggregation: exponentially weighted mean over the rolling window, with
        more recent articles weighted more heavily than older ones.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV dataframe with a DatetimeTZAware index (from yfinance).

        Columns added
        -------------
        news_sentiment_score    Exponentially-weighted mean sentiment in [-1, 1].
                                Positive = net bullish news flow, negative = bearish.
        news_article_count      Number of articles in the rolling window for that bar.
        news_bull               news_sentiment_score > bull_threshold (0.15 default)
        news_bear               news_sentiment_score < bear_threshold (-0.15 default)
        news_sentiment_strong_bull   score > 0.4  — high-conviction bullish headlines
        news_sentiment_strong_bear   score < -0.4 — high-conviction bearish headlines
        """
        news = self.fetch_and_score()

        # Ensure the OHLCV index is tz-aware UTC for alignment
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")

        # Default to zero if no news
        sentiment_score = np.zeros(len(df), dtype=float)
        article_count   = np.zeros(len(df), dtype=int)

        if not news.empty:
            news_times = news["datetime"].values          # numpy datetime64[ns, UTC]
            news_scores = news["sentiment_score"].values  # float array

            bar_times  = idx.values   # numpy datetime64 array

            # For each bar, find all articles published before it (no lookahead)
            # and compute an exponentially-weighted mean (half-life = 4 hours on 5m bars)
            # 4 hours = 48 five-minute bars → lambda = 1 - exp(-ln2/48) ≈ 0.0142
            decay = 1.0 - np.exp(-np.log(2) / 48)

            for i, bar_t in enumerate(bar_times):
                # Articles published strictly before this bar
                mask = news_times < bar_t
                if not mask.any():
                    continue

                art_scores = news_scores[mask]
                art_times  = news_times[mask]

                # Time deltas in 5-minute units (most recent = smallest delta)
                deltas = (bar_t - art_times).astype("timedelta64[m]").astype(float) / 5.0

                # Exponential weights: more recent → higher weight
                weights = np.exp(-decay * deltas)
                weights = weights / weights.sum()

                sentiment_score[i] = np.dot(weights, art_scores)
                article_count[i]   = int(mask.sum())

        df = df.copy()
        df["news_sentiment_score"]      = sentiment_score
        df["news_article_count"]        = article_count.astype(float)
        df["news_bull"]                 = (sentiment_score >  self.bull_threshold).astype(float)
        df["news_bear"]                 = (sentiment_score <  self.bear_threshold).astype(float)
        df["news_sentiment_strong_bull"]= (sentiment_score >  0.4).astype(float)
        df["news_sentiment_strong_bear"]= (sentiment_score < -0.4).astype(float)

        n_bull = int(df["news_bull"].sum())
        n_bear = int(df["news_bear"].sum())
        print(f"[sentiment] Features added. Bull bars: {n_bull}  Bear bars: {n_bear}  "
              f"Neutral: {len(df) - n_bull - n_bear}")

        return df


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",  default="SPY", help="Stock ticker (default: SPY)")
    p.add_argument("--days",    type=int, default=7)
    p.add_argument("--api_key", default=None)
    return p.parse_args()


if __name__ == "__main__":
    import yfinance as yf
    args = parse_args()

    pipe = SentimentPipeline(
        ticker    = args.ticker,
        api_key   = args.api_key,
        days_back = args.days + 5,   # a few extra days buffer
    )

    df = yf.download(args.ticker, period=f"{args.days}d", interval="5m",
                     progress=False, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df.index   = pd.to_datetime(df.index)

    df = pipe.add_sentiment_features(df)

    print("\nSample (last 10 bars):")
    print(df[["Close", "news_sentiment_score", "news_article_count",
              "news_bull", "news_bear"]].tail(10).to_string())

    # Show the raw scored headlines
    news = pipe.fetch_and_score()
    if not news.empty:
        print(f"\nTop 5 most bullish headlines:")
        print(news.nlargest(5, "sentiment_score")[["datetime","headline","sentiment_score"]].to_string(index=False))
        print(f"\nTop 5 most bearish headlines:")
        print(news.nsmallest(5, "sentiment_score")[["datetime","headline","sentiment_score"]].to_string(index=False))
