from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from data_utils import load_city_catalog, load_city_series, regression_metrics
from inference import (
    IncompatibleCheckpointError,
    find_artifact_dir,
    forecast_future,
    load_trained_bundle,
)
from train import train_city


PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
}

HOVER_LINE = (
    "<b>%{fullData.name}</b><br>"
    "Date: %{x|%Y-%m-%d}<br>"
    "ZHVI: $%{y:,.0f}<extra></extra>"
)

HOVER_BAR = (
    "Month: %{x}<br>"
    "Mean ZHVI: $%{y:,.0f}<extra></extra>"
)

HOVER_FORE_HISTORY = (
    "<b>History</b><br>"
    "Date: %{x|%Y-%m-%d}<br>"
    "ZHVI: $%{y:,.0f}<extra></extra>"
)

HOVER_FORE_FCAST = (
    "<b>Forecast</b><br>"
    "Date: %{x|%Y-%m-%d}<br>"
    "ZHVI: $%{y:,.0f}<extra></extra>"
)


def show_plotly(fig: go.Figure) -> None:
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def plot_test_actual_vs_pred(
    td: pd.DatetimeIndex | pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> go.Figure:
    t = pd.to_datetime(td)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_true,
            mode="lines",
            name="Actual",
            line=dict(color="steelblue", width=2),
            hovertemplate=HOVER_LINE,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_pred,
            mode="lines",
            name="Predicted",
            line=dict(color="crimson", width=2),
            hovertemplate=HOVER_LINE,
        )
    )
    fig.update_layout(
        title=dict(
            text="Test window: actual vs predicted ZHVI",
            x=0.5,
            xanchor="center",
            pad=dict(t=10, b=44),
        ),
        template="plotly_white",
        margin=dict(l=45, r=25, t=115, b=45),
        xaxis_title="Date",
        yaxis_title="ZHVI (USD)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.9)",
        ),
    )
    return fig


st.set_page_config(
    page_title="House Price LSTM (Zillow)",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_catalog(csv_path: str) -> pd.DataFrame:
    return load_city_catalog(csv_path)


@st.cache_data(show_spinner=False)
def cached_series(csv_path: str, region_name: str, state: str) -> pd.DataFrame:
    return load_city_series(csv_path, region_name, state)


def main() -> None:
    st.title("House price prediction (LSTM)")
    st.caption("Zillow ZHVI monthly series — train/test split, Adam, RMSE / MAE / R², forecast.")

    path = cfg.zillow_csv_path()
    if not path.exists():
        st.error(
            f"Zillow CSV not found: `{path}`. Set `DEFAULT_DATA_PATH` in `config.py`, "
            "or set environment variable `ZILLOW_CSV` to the full path."
        )
        return

    catalog = cached_catalog(str(path))
    search = st.sidebar.text_input("Search city", value="New York").strip().lower()
    if search:
        mask = catalog["label"].str.lower().str.contains(search, na=False)
        choices = catalog.loc[mask]
    else:
        choices = catalog.head(200)
    if choices.empty:
        st.warning("No cities match that search.")
        return
    display = choices.head(300)
    labels = display["label"].tolist()
    pick = st.sidebar.selectbox("Select city", labels, index=0)
    row = display[display["label"] == pick].iloc[0]
    region_name = str(row["RegionName"])
    state = str(row["State"])

    tab_hist, tab_train, tab_fore = st.tabs(
        ["Historical & EDA", "Train & evaluate", "Forecast"]
    )

    series = cached_series(str(path), region_name, state)

    with tab_hist:
        st.subheader(f"{region_name}, {state}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Trend**")
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    x=series["date"],
                    y=series["price"],
                    mode="lines",
                    name="ZHVI",
                    line=dict(color="steelblue", width=2),
                    hovertemplate=HOVER_LINE,
                )
            )
            fig1.update_layout(
                template="plotly_white",
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis_title="Date",
                yaxis_title="ZHVI (USD)",
                hovermode="x unified",
                showlegend=False,
            )
            show_plotly(fig1)
        with c2:
            st.markdown("**Seasonality** — mean ZHVI by calendar month")
            tmp = series.copy()
            tmp["month"] = tmp["date"].dt.month
            by_m = tmp.groupby("month", as_index=False)["price"].mean()
            fig2 = go.Figure()
            fig2.add_trace(
                go.Bar(
                    x=by_m["month"],
                    y=by_m["price"],
                    marker_color="darkseagreen",
                    hovertemplate=HOVER_BAR,
                    name="Mean ZHVI",
                )
            )
            fig2.update_layout(
                template="plotly_white",
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis_title="Month (1–12)",
                yaxis_title="Mean ZHVI",
                xaxis=dict(tickmode="linear", dtick=1),
                hovermode="closest",
                showlegend=False,
            )
            show_plotly(fig2)
        st.markdown(
            "**City comparison** — use search to open another city in a second browser tab."
        )

    with tab_train:
        st.markdown(
            "Chronological **train/test split** · **Univariate** input: **log1p(ZHVI)** only, "
            "**StandardScaled** on the train window · **One-step-ahead** target, then **inverse** + **expm1** · "
            "**Single-layer LSTM** · **Adam** + **MSE** (fixed epoch count, no LR schedule or early stopping)."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            lookback = st.number_input("Lookback (months)", 12, 180, cfg.DEFAULT_LOOKBACK)
        with col_b:
            epochs = st.number_input("Epochs", 10, 400, cfg.DEFAULT_EPOCHS)
        with col_c:
            hidden = st.number_input("LSTM hidden size", 16, 256, cfg.DEFAULT_HIDDEN)

        def render_test_metrics_and_plot(
            td: pd.DatetimeIndex | pd.Series,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            *,
            heading: str,
        ) -> None:
            m = regression_metrics(y_true, y_pred)
            st.markdown(f"#### {heading}")
            st.caption("Metrics are on the **held-out test** window (one-step-ahead predictions).")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"${m['rmse']:,.0f}", help="Root mean squared error in USD.")
            c2.metric("MAE", f"${m['mae']:,.0f}", help="Mean absolute error in USD.")
            c3.metric("R²", f"{m['r2']:.4f}", help="Coefficient of determination (1 is best).")
            show_plotly(plot_test_actual_vs_pred(td, y_true, y_pred))

        if st.button("Train / refresh model", type="primary"):
            with st.spinner("Training…"):
                try:
                    out = train_city(
                        csv_path=path,
                        region_name=region_name,
                        state=state,
                        lookback=int(lookback),
                        hidden=int(hidden),
                        epochs=int(epochs),
                    )
                except Exception as e:
                    st.exception(e)
                    return
            st.success(f"Saved to `{out['artifact_dir']}`")
            render_test_metrics_and_plot(
                pd.to_datetime(out["test_dates"]),
                out["y_test_inv"],
                out["preds_inv"],
                heading="Test set (this run)",
            )
        else:
            art = find_artifact_dir(region_name, state)
            if art and (art / "meta.json").exists():
                meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
                st.info(f"Found existing model in `{art}`.")
                npz_path = art / "test_predictions.npz"
                if npz_path.exists():
                    z = np.load(npz_path)
                    td = pd.to_datetime(z["dates_ns"].astype("datetime64[ns]"))
                    yt = z["y_true"]
                    yp = z["y_pred"]
                    render_test_metrics_and_plot(
                        td,
                        yt,
                        yp,
                        heading="Test set (saved run)",
                    )
            else:
                st.info("No trained model for this city yet. Adjust hyperparameters and click **Train / refresh model**.")

    with tab_fore:
        st.markdown(
            "**Recursive multi-step forecast:** each predicted value updates the **log-price window** "
            "for the next step. Choose how many future months to show (1–36)."
        )
        art = find_artifact_dir(region_name, state)
        if not art:
            st.warning("Train the model first (Train & evaluate tab).")
            return
        try:
            model, scaler, meta = load_trained_bundle(art)
        except IncompatibleCheckpointError as e:
            st.error(str(e))
            st.caption(
                "Tip: stop this Streamlit app if Windows says the folder is in use, then delete the folder "
                "or use the button below."
            )
            if st.button(
                "Remove incompatible checkpoint folder",
                key=f"rm_incompat_ckpt_{art.name}",
                help="Deletes saved weights for this city so you can train a fresh one-step model.",
            ):
                try:
                    shutil.rmtree(art)
                    st.success("Removed. Retraining on the Train tab will create a new checkpoint.")
                    st.rerun()
                except OSError as err:
                    st.warning(
                        f"Could not delete `{art}`: {err}. Close anything using those files, "
                        "delete the folder in File Explorer, then refresh this page."
                    )
            return
        except Exception as e:
            st.exception(e)
            return
        lb = int(meta.get("lookback", model.lookback))
        if len(series) < lb:
            st.error("Not enough history for this model's lookback window.")
            return
        months = st.slider("Months ahead", 1, 36, 12)
        preds = forecast_future(
            model,
            scaler,
            series["price"].to_numpy(dtype=np.float64),
            series["date"],
            int(months),
            meta=meta,
        )
        last_date = pd.Timestamp(series["date"].iloc[-1])
        fut_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=len(preds),
            freq="MS",
        )

        st.markdown("**History and forecast** (hover for values)")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series["date"],
                y=series["price"],
                mode="lines",
                name="History",
                line=dict(color="steelblue", width=2),
                hovertemplate=HOVER_FORE_HISTORY,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fut_dates,
                y=preds,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="crimson", width=2),
                marker=dict(size=6),
                hovertemplate=HOVER_FORE_FCAST,
            )
        )
        fig.update_layout(
            title=dict(
                text="Historical ZHVI and LSTM forecast",
                x=0.5,
                xanchor="center",
                pad=dict(b=28),
            ),
            template="plotly_white",
            margin=dict(l=40, r=20, t=100, b=40),
            xaxis_title="Date",
            yaxis_title="ZHVI",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0.5,
                xanchor="center",
            ),
        )
        show_plotly(fig)

        st.dataframe(
            pd.DataFrame({"month": fut_dates, "predicted_zhvi": preds}),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
