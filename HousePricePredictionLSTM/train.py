from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from data_utils import (
    FEATURE_COLUMNS,
    load_city_series,
    regression_metrics,
    scaled_log_price_to_usd,
    sequences_from_series,
)
from inference import artifact_tag
from model_lstm import HousePriceLSTM, get_device


PREPROCESS_TAG = "log1p_standard"
RESIDUAL_READOUT = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_city(
    csv_path: Path,
    region_name: str,
    state: str,
    lookback: int = cfg.DEFAULT_LOOKBACK,
    hidden: int = cfg.DEFAULT_HIDDEN,
    num_layers: int = cfg.DEFAULT_NUM_LAYERS,
    dropout: float = cfg.DEFAULT_DROPOUT,
    epochs: int = cfg.DEFAULT_EPOCHS,
    batch_size: int = cfg.DEFAULT_BATCH_SIZE,
    lr: float = cfg.DEFAULT_LR,
    train_ratio: float = cfg.DEFAULT_TRAIN_RATIO,
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    df = load_city_series(csv_path, region_name, state)
    prices = df["price"].to_numpy(dtype=np.float64)
    dates = df["date"]

    x_train, y_train, x_test, y_test, scaler, train_seq_count = sequences_from_series(
        prices, dates, lookback, train_ratio
    )
    n_input = int(scaler.n_features_in_)

    device = get_device()
    model = HousePriceLSTM(
        lookback=lookback,
        input_size=n_input,
        hidden_size=hidden,
        num_layers=num_layers,
        dropout=dropout,
        residual_readout=RESIDUAL_READOUT,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    fit_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in fit_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x_test).to(device)
        preds_scaled = model(xt).cpu().numpy()

    y_test_inv = scaled_log_price_to_usd(scaler, y_test).ravel()
    preds_inv = scaled_log_price_to_usd(scaler, preds_scaled).ravel()
    metrics = regression_metrics(y_test_inv, preds_inv)

    split_time_idx = train_seq_count + lookback
    test_dates = dates.iloc[split_time_idx : split_time_idx + len(y_test)].reset_index(
        drop=True
    )

    tag = artifact_tag(region_name, state)
    out_dir = cfg.ARTIFACTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "lookback": lookback,
            "hidden": hidden,
            "num_layers": num_layers,
            "dropout": dropout,
            "residual_readout": RESIDUAL_READOUT,
            "input_size": n_input,
            "feature_columns": FEATURE_COLUMNS,
            "region_name": region_name,
            "state": state,
            "preprocess": PREPROCESS_TAG,
        },
        out_dir / "checkpoint.pt",
    )
    joblib.dump(scaler, out_dir / "scaler.joblib")

    meta = {
        "region_name": region_name,
        "state": state,
        "lookback": lookback,
        "hidden": hidden,
        "num_layers": num_layers,
        "dropout": dropout,
        "residual_readout": RESIDUAL_READOUT,
        "input_size": n_input,
        "feature_columns": FEATURE_COLUMNS,
        "preprocess": PREPROCESS_TAG,
        "train_ratio": train_ratio,
        "epochs_ran": epochs,
        "epochs_cap": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "artifact_tag": tag,
        "metrics": metrics,
        "csv_path": str(Path(csv_path).resolve()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    dates_ns = test_dates.values.astype("datetime64[ns]").astype(np.int64)
    np.savez(
        out_dir / "test_predictions.npz",
        dates_ns=dates_ns,
        y_true=y_test_inv,
        y_pred=preds_inv,
    )

    return {
        "artifact_dir": str(out_dir),
        "metrics": metrics,
        "meta": meta,
        "test_dates": test_dates,
        "y_test_inv": y_test_inv,
        "preds_inv": preds_inv,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train LSTM on Zillow ZHVI for one city.")
    p.add_argument("--data", type=Path, default=cfg.DEFAULT_DATA_PATH)
    p.add_argument("--city", required=True, help="RegionName, e.g. 'New York'")
    p.add_argument("--state", required=True, help="Two-letter state, e.g. 'NY'")
    p.add_argument("--lookback", type=int, default=cfg.DEFAULT_LOOKBACK)
    p.add_argument("--hidden", type=int, default=cfg.DEFAULT_HIDDEN)
    p.add_argument("--num-layers", type=int, default=cfg.DEFAULT_NUM_LAYERS)
    p.add_argument("--dropout", type=float, default=cfg.DEFAULT_DROPOUT)
    p.add_argument("--epochs", type=int, default=cfg.DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.DEFAULT_LR)
    p.add_argument("--train-ratio", type=float, default=cfg.DEFAULT_TRAIN_RATIO)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.data.exists():
        raise SystemExit(f"CSV not found: {args.data}")

    result = train_city(
        csv_path=args.data,
        region_name=args.city,
        state=args.state,
        lookback=args.lookback,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    m = result["metrics"]
    print("Saved:", result["artifact_dir"])
    print(f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()
