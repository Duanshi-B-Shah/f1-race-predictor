"""Streamlit demo app for F1 race position prediction."""

from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "race_results.csv"

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="centered")
st.title("🏎️ F1 Race Position Predictor")
st.markdown("Predict a driver's finishing position based on race context and historical performance.")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return model, feature_cols


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model, feature_cols = load_model()
    df = load_data()
except FileNotFoundError:
    st.error("Model or data not found. Run `python3 src/train.py` first.")
    st.stop()

# Build display name mappings
driver_name_map = (
    df.dropna(subset=["driver_name"])
    .drop_duplicates(subset=["driver_id"], keep="last")
    .set_index("driver_id")["driver_name"].to_dict()
)
circuit_name_map = (
    df.dropna(subset=["circuit_name"])
    .drop_duplicates(subset=["circuit_id"], keep="last")
    .set_index("circuit_id")["circuit_name"].to_dict()
)
constructor_name_map = (
    df.dropna(subset=["constructor_name"])
    .drop_duplicates(subset=["constructor_id"], keep="last")
    .set_index("constructor_id")["constructor_name"].to_dict()
)

# Sorted display lists
driver_ids = sorted(driver_name_map.keys(), key=lambda d: driver_name_map[d])
circuit_ids = sorted(circuit_name_map.keys(), key=lambda c: circuit_name_map[c])

col1, col2 = st.columns(2)
with col1:
    selected_driver = st.selectbox(
        "Driver", driver_ids,
        format_func=lambda d: driver_name_map.get(d, d),
    )
with col2:
    selected_circuit = st.selectbox(
        "Circuit", circuit_ids,
        format_func=lambda c: circuit_name_map.get(c, c),
    )

grid_position = st.slider("Grid Position", 1, 20, 5)

# Show current team
driver_races = df[df["driver_id"] == selected_driver].sort_values(["season", "round"])
if len(driver_races) > 0:
    latest = driver_races.iloc[-1]
    team_name = constructor_name_map.get(latest["constructor_id"], latest["constructor_id"])
    st.caption(f"Team: {team_name}")

# Compute features from historical data
recent_races = driver_races.tail(5)
constructor_id = recent_races["constructor_id"].iloc[-1] if len(recent_races) > 0 else None
constructor_races = df[df["constructor_id"] == constructor_id].tail(5) if constructor_id else pd.DataFrame()
circuit_races = df[(df["driver_id"] == selected_driver) & (df["circuit_id"] == selected_circuit)]

driver_avg = recent_races["finish_position"].dropna().mean() if len(recent_races) > 0 else 10.0
constructor_avg = constructor_races["finish_position"].dropna().mean() if len(constructor_races) > 0 else 10.0
circuit_avg = circuit_races["finish_position"].dropna().mean() if len(circuit_races) > 0 else driver_avg

dnf_count = recent_races["status"].apply(lambda s: 0 if s == "Finished" else 1).sum()
dnf_rate = dnf_count / max(len(recent_races), 1)

pos_changes = recent_races["grid_position"] - recent_races["finish_position"].dropna()
avg_pos_change = pos_changes.mean() if len(pos_changes) > 0 else 0.0

features = pd.DataFrame([{
    "grid_position": grid_position,
    "driver_avg_finish": driver_avg,
    "constructor_avg_finish": constructor_avg,
    "circuit_driver_avg": circuit_avg,
    "driver_dnf_rate": dnf_rate,
    "grid_position_change": avg_pos_change,
}])

if st.button("Predict Finish Position", type="primary"):
    prediction = model.predict(features[feature_cols])[0]
    predicted_pos = max(1, round(prediction))

    st.metric("Predicted Finish Position", f"P{predicted_pos}")

    delta = grid_position - predicted_pos
    if delta > 0:
        st.success(f"Model predicts gaining ~{delta} position(s) from grid")
    elif delta < 0:
        st.warning(f"Model predicts losing ~{abs(delta)} position(s) from grid")
    else:
        st.info("Model predicts finishing roughly where they start")

    with st.expander("Feature Values Used"):
        st.dataframe(features.T.rename(columns={0: "Value"}))

# --- Race History Section (Interactive Plotly Chart) ---
st.divider()
driver_display = driver_name_map.get(selected_driver, selected_driver)
st.subheader(f"📊 Race History: {driver_display}")

history = driver_races.copy()
if len(history) > 0:
    history = history.sort_values(["season", "round"], ascending=True).reset_index(drop=True)
    chart_df = history.copy()
    chart_df["circuit_label"] = chart_df["circuit_id"].map(lambda c: circuit_name_map.get(c, c))
    chart_df["race_label"] = chart_df["season"].astype(str) + " R" + chart_df["round"].astype(str)
    chart_df["finish_position"] = pd.to_numeric(chart_df["finish_position"], errors="coerce")
    chart_df["grid_position"] = pd.to_numeric(chart_df["grid_position"], errors="coerce")

    plot_df = chart_df.dropna(subset=["finish_position", "grid_position"]).copy()
    plot_df["positions_gained"] = plot_df["grid_position"] - plot_df["finish_position"]

    if len(plot_df) > 0:
        fig = go.Figure()

        # Vertical bars: grid -> finish (green = gained, red = lost)
        for _, row in plot_df.iterrows():
            color = "#00c853" if row["finish_position"] <= row["grid_position"] else "#e10600"
            fig.add_trace(go.Scatter(
                x=[row["race_label"], row["race_label"]],
                y=[row["grid_position"], row["finish_position"]],
                mode="lines",
                line=dict(color=color, width=3),
                opacity=0.4,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Grid position line
        fig.add_trace(go.Scatter(
            x=plot_df["race_label"],
            y=plot_df["grid_position"],
            mode="lines+markers",
            name="Grid",
            line=dict(color="#888888", width=1.5, dash="dot"),
            marker=dict(size=7, symbol="square", color="#888888"),
            hovertemplate="<b>%{customdata[0]}</b><br>Grid: P%{y}<extra></extra>",
            customdata=plot_df[["circuit_label"]].values,
        ))

        # Finish position line
        fig.add_trace(go.Scatter(
            x=plot_df["race_label"],
            y=plot_df["finish_position"],
            mode="lines+markers",
            name="Finish",
            line=dict(color="#00d2be", width=3),
            marker=dict(size=9, color="#00d2be"),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Finish: P%{y}<br>"
                "Positions gained: %{customdata[1]:+d}"
                "<extra></extra>"
            ),
            customdata=plot_df[["circuit_label", "positions_gained"]].values,
        ))

        # Podium markers
        podiums = plot_df[plot_df["finish_position"] <= 3]
        if len(podiums) > 0:
            fig.add_trace(go.Scatter(
                x=podiums["race_label"],
                y=podiums["finish_position"],
                mode="markers",
                name="Podium",
                marker=dict(size=14, color="#FFD700", symbol="star",
                            line=dict(width=1, color="#333")),
                hovertemplate="<b>🏆 Podium</b><br>%{customdata[0]}<br>P%{y}<extra></extra>",
                customdata=podiums[["circuit_label"]].values,
            ))

        # Layout — P1 at top, clean Y-axis with only integer ticks 1-20
        y_max = int(max(plot_df["grid_position"].max(), plot_df["finish_position"].max()))
        fig.update_layout(
            yaxis=dict(
                autorange="reversed",
                title="Position",
                range=[0.5, min(y_max + 1, 21)],
                dtick=1,
                gridcolor="rgba(255,255,255,0.07)",
            ),
            xaxis=dict(
                title="",
                tickangle=-45,
                gridcolor="rgba(255,255,255,0.05)",
            ),
            template="plotly_dark",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=12),
            ),
            margin=dict(l=50, r=20, t=40, b=80),
            height=420,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Compact table in expander
    with st.expander("Full Race Results Table", expanded=False):
        table_df = history[["season", "round", "circuit_id", "grid_position", "finish_position", "status"]].copy()
        table_df = table_df.sort_values(["season", "round"], ascending=False).reset_index(drop=True)
        table_df["circuit_id"] = table_df["circuit_id"].map(lambda c: circuit_name_map.get(c, c))
        table_df["grid_position"] = table_df["grid_position"].astype(int)
        table_df["finish_position"] = pd.to_numeric(table_df["finish_position"], errors="coerce").fillna(0).astype(int)
        table_df.columns = ["Season", "Round", "Circuit", "Grid", "Finish", "Status"]

        def highlight_result(row):
            styles = [""] * len(row)
            try:
                grid = int(row["Grid"])
                finish = int(row["Finish"])
                if finish < grid:
                    styles[4] = "color: #00c853"
                elif finish > grid:
                    styles[4] = "color: #ff1744"
            except (ValueError, TypeError):
                pass
            if row["Status"] != "Finished":
                styles[5] = "color: #ff9100"
            return styles

        st.dataframe(
            table_df.style.apply(highlight_result, axis=1),
            use_container_width=True,
            height=min(400, 35 * len(table_df) + 38),
        )

    st.caption(f"{len(history)} race(s) on record")
else:
    st.info("No race history available for this driver.")

st.divider()
st.caption("Data: OpenF1 API (2023–2026) | Model: XGBoost Regressor")
