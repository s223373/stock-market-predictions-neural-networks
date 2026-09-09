from dataset_builder import _is_binary_col, FEATURES
from feature_engineering import build_features
import yfinance as yf
import pandas as pd

raw = yf.download("SPY", period="7d", interval="1m", progress=False, auto_adjust=True)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
enriched = build_features(raw, period="7d", interval="1m", ticker="SPY")

print("enriched shape:", enriched.shape)
print("has isGreen col:", "isGreen" in enriched.columns)

checked = 0
failed = 0
for c in FEATURES:
    if c == "Close" or c not in enriched.columns:
        continue
    checked += 1
    if not _is_binary_col(enriched[c]):
        failed += 1
        vals = pd.to_numeric(enriched[c], errors="coerce").dropna().unique()
        print(f"FAIL {c:35s} dtype={enriched[c].dtype} n_unique={len(vals)} sample={vals[:6]}")

print(f"\nchecked={checked} failed={failed}")