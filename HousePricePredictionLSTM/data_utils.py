from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


DATE_COL_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Single supervised channel: scaled log1p(ZHVI).
FEATURE_COLUMNS = ["log1p_price"]


def date_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if DATE_COL_PATTERN.match(str(c))]


def load_city_catalog(csv_path: str | pd.PathLike) -> pd.DataFrame:
    """Distinct cities sorted by Zillow SizeRank (smaller is larger market)."""
    df = pd.read_csv(
        csv_path,
        usecols=["RegionName", "State", "SizeRank"],
        low_memory=False,
    )
    df = df.groupby(["RegionName", "State"], as_index=False)["SizeRank"].min()
    df = df.sort_values("SizeRank", ascending=True)
    df["label"] = df["RegionName"] + ", " + df["State"]
    return df.reset_index(drop=True)


def load_city_series(
    csv_path: str | pd.PathLike,
    region_name: str,
    state: str,
) -> pd.DataFrame:
    """Load one city's monthly ZHVI series as a long-format DataFrame: date, price."""
    df = pd.read_csv(csv_path, low_memory=False)
    if "RegionName" not in df.columns or "State" not in df.columns:
        raise ValueError("Expected RegionName and State columns in Zillow city CSV.")

    row = df[(df["RegionName"] == region_name) & (df["State"] == state)]
    if row.empty:
        raise ValueError(f"No row found for city={region_name!r}, state={state!r}")

    dcols = date_columns(df)
    if not dcols:
        raise ValueError("No YYYY-MM-DD date columns found (wide Zillow format).")

    values = row.iloc[0][dcols].astype(float)
    long = (
        pd.DataFrame({"date": pd.to_datetime(dcols), "price": values.values})
        .sort_values("date")
        .reset_index(drop=True)
    )
    long = long.dropna(subset=["price"])
    long["price"] = long["price"].interpolate(limit_direction="both")
    long = long.dropna(subset=["price"])
    if long.empty:
        raise ValueError("Series is empty after handling missing values.")
    return long


def time_series_train_test_split(
    n: int, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    split = int(np.floor(n * train_ratio))
    split = max(split, 1)
    split = min(split, n - 1)
    train_idx = np.arange(0, split)
    test_idx = np.arange(split, n)
    return train_idx, test_idx


def _build_sequences_univariate(
    scaled_1d: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """scaled_1d: (T,) next-step target. Returns X (N, lookback, 1), y (N, 1)."""
    s = np.asarray(scaled_1d, dtype=np.float32).ravel()
    t = len(s)
    if t <= lookback:
        raise ValueError(f"Need T > lookback; got T={t}, lookback={lookback}.")
    x_list, y_list = [], []
    for i in range(lookback, t):
        x_list.append(s[i - lookback : i])
        y_list.append(s[i])
    x = np.asarray(x_list, dtype=np.float32).reshape(len(x_list), lookback, 1)
    y = np.asarray(y_list, dtype=np.float32)[:, np.newaxis]
    return x, y


def scaled_log_price_to_usd(scaler: StandardScaler, y_scaled: np.ndarray) -> np.ndarray:
    """Invert scaled log1p price to USD (works for 1D target with single-column scaler)."""
    arr = np.asarray(y_scaled, dtype=np.float64)
    orig_shape = arr.shape
    flat = arr.ravel()
    n = flat.shape[0]
    f = int(scaler.n_features_in_)
    pad = np.zeros((n, f), dtype=np.float64)
    pad[:, 0] = flat
    logp = scaler.inverse_transform(pad)[:, 0]
    out = np.expm1(logp)
    return out.reshape(orig_shape)


def sequences_from_series(
    prices: np.ndarray,
    dates: pd.DatetimeIndex | pd.Series,
    lookback: int,
    train_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
    """
    Univariate: log1p(ZHVI), StandardScaler fit on train rows only, one-step-ahead windows.
    `dates` is kept for API compatibility and test-date alignment (same length as prices).
    """
    p = np.asarray(prices, dtype=np.float64).ravel()
    di = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(di) != len(p):
        raise ValueError("dates must align with prices (same length).")
    logp = np.log1p(p).reshape(-1, 1)
    n = len(logp)
    train_idx, _ = time_series_train_test_split(n, train_ratio)
    split = len(train_idx)
    if split <= lookback:
        raise ValueError(
            "Train window is too short for this lookback; reduce lookback or train_ratio."
        )
    scaler = StandardScaler()
    scaler.fit(logp[train_idx])
    scaled = scaler.transform(logp).astype(np.float32).ravel()
    x, y = _build_sequences_univariate(scaled, lookback)
    train_seq_count = split - lookback
    if train_seq_count <= 0 or train_seq_count > len(x):
        raise ValueError("Could not form train sequences; adjust lookback or train_ratio.")
    x_train, y_train = x[:train_seq_count], y[:train_seq_count]
    x_test, y_test = x[train_seq_count:], y[train_seq_count:]
    if len(x_test) == 0:
        raise ValueError("No test sequences; try lowering train_ratio or lookback.")
    return x_train, y_train, x_test, y_test, scaler, train_seq_count


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}
