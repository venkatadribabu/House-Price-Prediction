from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import config as cfg
from model_lstm import HousePriceLSTM, get_device


class IncompatibleCheckpointError(ValueError):
    """Weights on disk don't match the current model (e.g. old 12-step direct head)."""


def _readout_output_dim(state: dict[str, Any]) -> int | None:
    w = state.get("dense.weight")
    if w is None:
        return None
    return int(w.shape[0])


def assert_checkpoint_matches_current_model(
    state: dict[str, Any], artifact_dir: Path | None = None
) -> None:
    """Current app expects one-step readout (dense out_features == 1)."""
    out = _readout_output_dim(state)
    if out is not None and out != 1:
        loc = f"`{artifact_dir}`" if artifact_dir is not None else "this city's artifact folder"
        raise IncompatibleCheckpointError(
            f"This folder was saved with a {out}-output readout (older direct multi-step setup). "
            "The app now uses a one-step LSTM (1 output). Open Train & evaluate and click "
            f"Train / refresh model to save a new checkpoint, or delete {loc} and train again."
        )


def artifact_tag(region_name: str, state: str) -> str:
    import hashlib

    key = f"{region_name.strip().lower()}|{state.strip().upper()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def artifact_dir_for_city(region_name: str, state: str) -> Path:
    return cfg.ARTIFACTS_DIR / artifact_tag(region_name, state)


def find_artifact_dir(region_name: str, state: str) -> Path | None:
    direct = artifact_dir_for_city(region_name, state)
    if (direct / "meta.json").exists():
        return direct
    if not cfg.ARTIFACTS_DIR.exists():
        return None
    for meta_path in cfg.ARTIFACTS_DIR.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("region_name") == region_name and meta.get("state") == state:
            return meta_path.parent
    return None


def infer_num_layers_from_state_dict(state_dict: dict[str, Any]) -> int:
    """PyTorch stacked LSTM uses keys like lstm.weight_ih_l0, lstm.weight_ih_l1, …"""
    mx = -1
    for k in state_dict:
        m = re.search(r"lstm\.weight_ih_l(\d+)$", k)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1 if mx >= 0 else 1


def infer_input_size_from_state_dict(state_dict: dict[str, Any]) -> int:
    w = state_dict.get("lstm.weight_ih_l0")
    if w is None:
        return 1
    return int(w.shape[1])


def load_trained_bundle(artifact_dir: Path) -> tuple[HousePriceLSTM, Any, dict[str, Any]]:
    path = artifact_dir / "checkpoint.pt"
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location="cpu")
    lookback = int(ck["lookback"])
    hidden = int(ck["hidden"])
    state = ck["model_state"]
    num_layers = infer_num_layers_from_state_dict(state)
    dropout = float(ck.get("dropout", 0.2 if num_layers > 1 else 0.0))
    residual_readout = bool(ck.get("residual_readout", False))
    input_size = int(ck.get("input_size", infer_input_size_from_state_dict(state)))

    assert_checkpoint_matches_current_model(state, artifact_dir)

    model = HousePriceLSTM(
        lookback=lookback,
        input_size=input_size,
        hidden_size=hidden,
        num_layers=num_layers,
        dropout=dropout,
        residual_readout=residual_readout,
    )
    model.load_state_dict(state)
    model.eval()
    scaler = joblib.load(artifact_dir / "scaler.joblib")
    meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
    return model, scaler, meta


def _uses_log1p_standard(scaler: Any, meta: dict[str, Any]) -> bool:
    if meta.get("preprocess") == "log1p_standard":
        return True
    return isinstance(scaler, StandardScaler)


@torch.no_grad()
def forecast_future(
    model: HousePriceLSTM,
    scaler: Any,
    prices_all: np.ndarray,
    dates_all: pd.DatetimeIndex | pd.Series,
    steps: int,
    meta: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Recursive multi-step forecast from log1p(price) scaled windows.
    `dates_all` must align in length with `prices_all` (used for validation only).
    """
    device = get_device()
    model = model.to(device)
    lookback = model.lookback
    meta = meta if meta is not None else {}
    prices = np.asarray(prices_all, dtype=np.float64).ravel()
    dates = pd.DatetimeIndex(pd.to_datetime(dates_all))
    if len(prices) != len(dates):
        raise ValueError("prices_all and dates_all must have the same length.")
    if len(prices) < lookback:
        raise ValueError(f"Need at least {lookback} points; got {len(prices)}.")

    n_feat = int(getattr(scaler, "n_features_in_", 1))
    if n_feat != 1:
        raise ValueError(
            "This build expects a univariate (1-feature) checkpoint. "
            "Retrain after removing multivariate features, or use an older artifact format."
        )

    hist = prices.copy()
    log_standard = _uses_log1p_standard(scaler, meta)
    if log_standard:
        feats = np.log1p(hist.reshape(-1, 1))
    else:
        feats = hist.reshape(-1, 1)
    scaled_full = scaler.transform(feats).astype(np.float32)
    window = scaled_full[-lookback:].copy()
    preds_scaled: list[float] = []
    for _ in range(steps):
        x = torch.from_numpy(window.reshape(1, lookback, 1)).to(device)
        nxt = float(model(x).cpu().numpy().reshape(-1)[0])
        preds_scaled.append(nxt)
        window = np.concatenate([window[1:], [[nxt]]], axis=0).astype(np.float32)
    arr = np.asarray(preds_scaled, dtype=np.float64).reshape(-1, 1)
    inv = scaler.inverse_transform(arr).ravel()
    if log_standard:
        return np.expm1(inv)
    return inv
