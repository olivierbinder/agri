"""Gradio UI for the Agri yield predictor, calling the FastAPI model server.

Layout is deliberately narrow: a single controls column on the left, results on the
right, so nothing stretches across a wide screen.
"""

import os

import gradio as gr
import pandas as pd
import requests

from agri.core import constants

# Set via the API_URL env var when this app calls a separately deployed API
# (Cloud Run injects it from the deploy job); defaults to local dev. `just app` uses
# API_PORT, so honour it too — otherwise the UI would call 8000 while the API listens
# elsewhere.
API_URL = os.environ.get(
    "API_URL", f"http://localhost:{os.environ.get('API_PORT', '8000')}"
)
API_TIMEOUT_SECONDS = 30

# Slider bounds, from the observed ranges in the dataset (see the EDA notebook):
# rain 51-3240 mm, pesticides 0.04-367778 t, temp 1.3-30.65 C.
RAINFALL_RANGE = (50, 3250, 10)
PESTICIDES_RANGE = (0, 370000, 100)
TEMP_RANGE = (1, 31, 0.5)

LABELS = {
    "average_rain_fall_mm_per_year": "Average Rainfall (mm/year)",
    "pesticides_tonnes": "Pesticides (tonnes)",
    "avg_temp": "Average Temperature (°C)",
}


# %% API


def call_api(endpoint: str, payload: dict) -> dict:
    """POST `payload` to `endpoint` and return the decoded JSON response."""
    try:
        response = requests.post(
            f"{API_URL}{endpoint}", json=payload, timeout=API_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        raise gr.Error(f"Could not reach the API at {API_URL}: {error}") from error
    return response.json()


def fetch_reference_conditions(area: str, item: str, year: int) -> dict | None:
    """Return the real recorded conditions for Area/Item/Year, or None if unknown."""
    data = call_api(
        "/predict",
        {
            "Area": area,
            "Item": item,
            "Year": year,
            "average_rain_fall_mm_per_year": constants.DEFAULT_RAINFALL,
            "pesticides_tonnes": constants.DEFAULT_PESTICIDES,
            "avg_temp": constants.DEFAULT_TEMP,
        },
    )
    return data["reference_conditions"]


# %% FORMATTING


def format_prediction(data: dict, year: int) -> str:
    """Render the prediction, and only compare to the actual when the conditions match."""
    unit = data["unit"]
    text = f"### Predicted Yield: {data['prediction']:,.2f} {unit}"
    if data["actual"] is not None:
        delta_pct = (data["prediction"] - data["actual"]) / data["actual"] * 100
        text += (
            f"\n\n**Actual yield in {year}:** {data['actual']:,.2f} {unit}  \n"
            f"_(model was {delta_pct:+.1f}% off)_"
        )
    elif data["reference_conditions"] is not None:
        real = " · ".join(
            f"{LABELS[feature]}: **{value:,.2f}**"
            for feature, value in data["reference_conditions"].items()
        )
        text += (
            f"\n\n_No comparison with the real {year} yield: your rainfall / pesticides /"
            f" temperature differ from the recorded conditions ({real}). "
            "Change them back to compare._"
        )
    return text


def format_recommendation(ranking: pd.DataFrame, unit: str) -> str:
    """Summarize the top-ranked crop and whether the ranking explains real yields."""
    best = ranking.iloc[0]
    summary = (
        f"### 🥇 Best Crop: {best['Item']}  \n"
        f"_{best['relative_score']:,.2f}x its usual yield → "
        f"{best['prediction']:,.2f} {unit}_"
    )
    if ranking["actual"].isna().all():
        summary += (
            "\n\n_No actual yields shown: the ranking uses the conditions above, which "
            "differ from the ones recorded for this area and year._"
        )
    return summary


def rank_table(ranking: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Rename ranking columns for display and drop the actual column when empty."""
    table = ranking.rename(
        columns={
            "Item": "Crop",
            "prediction": f"Predicted Yield ({unit})",
            "relative_score": "Relative Score",
            "actual": f"Actual Yield ({unit})",
        }
    )
    if table[f"Actual Yield ({unit})"].isna().all():
        table = table.drop(columns=[f"Actual Yield ({unit})"])
    return table


# %% CALLBACKS


def predict_yield(
    area: str, item: str, year: int, rainfall: float, pesticides: float, temp: float
) -> str:
    """Call /predict and render the result."""
    data = call_api(
        "/predict",
        {
            "Area": area,
            "Item": item,
            "Year": year,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temp,
        },
    )
    return format_prediction(data, year)


def recommend_crops(
    area: str, year: int, rainfall: float, pesticides: float, temp: float
) -> tuple[str, pd.DataFrame]:
    """Call /recommend and render the ranking."""
    data = call_api(
        "/recommend",
        {
            "Area": area,
            "Year": year,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temp,
        },
    )
    ranking = pd.DataFrame(data["recommendations"])
    return format_recommendation(ranking, data["unit"]), rank_table(
        ranking, data["unit"]
    )


def prefill_conditions(
    area: str, item: str, year: int
) -> tuple[float, float, float, gr.Slider]:
    """Pre-fill the climate sliders with the real recorded values, when known.

    Also updates the pesticides slider step: real values span 0.04-367778 t, so a fixed
    step either rounds small values to zero or makes large ones tedious to adjust.
    """
    conditions = fetch_reference_conditions(area, item, year)
    if conditions is None:
        values = (
            constants.DEFAULT_RAINFALL,
            constants.DEFAULT_PESTICIDES,
            constants.DEFAULT_TEMP,
        )
        step = PESTICIDES_RANGE[2]
    else:
        values = tuple(
            conditions[feature]
            for feature in (
                "average_rain_fall_mm_per_year",
                "pesticides_tonnes",
                "avg_temp",
            )
        )
        step = 0.01 if values[1] < 100 else max(1.0, round(values[1] / 1000))
    rainfall, pesticides, temp = values
    return rainfall, pesticides, temp, gr.Slider(step=step)


# %% LAYOUT

with gr.Blocks(title="Agri Yield Predictor") as demo:
    gr.Markdown(
        "# 🌾 Agri Yield Predictor\n"
        "Predicts crop yields from climate and agricultural data, via the FastAPI model "
        "server. The model is trained on **2008-2012**: 2013 is real held-out data "
        "(comparable), 2014-2015 are pure extrapolation."
    )

    with gr.Row(equal_height=False):
        # Left: all inputs, stacked in one narrow column.
        with gr.Column(scale=1, min_width=320):
            area = gr.Dropdown(
                constants.AREAS, value=constants.DEFAULT_AREA, label="Country / Area"
            )
            item = gr.Dropdown(
                constants.ITEMS, value=constants.DEFAULT_ITEM, label="Crop / Item"
            )
            year = gr.Slider(
                2013, 2015, value=constants.DEFAULT_YEAR, step=1, label="Year"
            )
            rainfall = gr.Slider(
                *RAINFALL_RANGE[:2],
                value=constants.DEFAULT_RAINFALL,
                step=RAINFALL_RANGE[2],
                label=LABELS["average_rain_fall_mm_per_year"],
            )
            pesticides = gr.Slider(
                *PESTICIDES_RANGE[:2],
                value=constants.DEFAULT_PESTICIDES,
                step=PESTICIDES_RANGE[2],
                label=LABELS["pesticides_tonnes"],
            )
            temp = gr.Slider(
                *TEMP_RANGE[:2],
                value=constants.DEFAULT_TEMP,
                step=TEMP_RANGE[2],
                label=LABELS["avg_temp"],
            )
            prefill_button = gr.Button("↺ Use the real recorded conditions")
            predict_button = gr.Button("🚀 Predict Yield", variant="primary")
            recommend_button = gr.Button("🏆 Recommend Best Crop", variant="primary")

        # Right: results only, so the eye doesn't travel the full screen width.
        with gr.Column(scale=2):
            with gr.Tab("🔮 Yield"):
                predict_output = gr.Markdown()
            with gr.Tab("🏆 Best crop"):
                recommend_summary = gr.Markdown()
                recommend_table = gr.Dataframe(label="Ranking", wrap=True)

    # Pre-fill whenever the identity of the plot changes: this is what makes the
    # predicted/actual comparison meaningful by default.
    prefill_inputs = [area, item, year]
    prefill_outputs = [rainfall, pesticides, temp, pesticides]
    for component in prefill_inputs:
        component.change(
            prefill_conditions, inputs=prefill_inputs, outputs=prefill_outputs
        )
    prefill_button.click(
        prefill_conditions, inputs=prefill_inputs, outputs=prefill_outputs
    )

    predict_button.click(
        predict_yield,
        inputs=[area, item, year, rainfall, pesticides, temp],
        outputs=predict_output,
    )
    recommend_button.click(
        recommend_crops,
        inputs=[area, year, rainfall, pesticides, temp],
        outputs=[recommend_summary, recommend_table],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
