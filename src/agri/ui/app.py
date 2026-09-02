import os

import pandas as pd
import requests
import streamlit as st

from agri.core import constants

st.set_page_config(page_title="Agri Yield Prediction", page_icon="🌾")

st.title("🌾 Agri Yield Predictor")
st.markdown(
    "This application predicts crop yields based on climate and agricultural data."
)

# Defaults to local dev; set via Streamlit secrets (st.secrets["API_URL"]) or the
# API_URL env var when this app calls a separately deployed API.
API_URL = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))
API_TIMEOUT_SECONDS = 30

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "What do you want to do?",
    ["🔮 Predict a yield", "🏆 Recommend the best crop"],
)

st.sidebar.header("Plot Context")
area = st.sidebar.selectbox(
    "Country / Area",
    constants.AREAS,
    index=constants.AREAS.index(constants.DEFAULT_AREA),
)

item = None
if mode == "🔮 Predict a yield":
    item = st.sidebar.selectbox(
        "Crop / Item",
        constants.ITEMS,
        index=constants.ITEMS.index(constants.DEFAULT_ITEM),
    )

year = st.sidebar.slider(
    "Year", min_value=1990, max_value=2050, value=constants.DEFAULT_YEAR, step=1
)
rainfall = st.sidebar.slider(
    "Average Rainfall (mm/year)",
    min_value=0.0,
    max_value=3000.0,
    value=constants.DEFAULT_RAINFALL,
    step=10.0,
)
pesticides = st.sidebar.slider(
    "Pesticides (tonnes)",
    min_value=0.0,
    max_value=2000.0,
    value=constants.DEFAULT_PESTICIDES,
    step=10.0,
)
temp = st.sidebar.slider(
    "Average Temperature (°C)",
    min_value=-20.0,
    max_value=50.0,
    value=constants.DEFAULT_TEMP,
    step=0.5,
)

st.write("### Selected Context")
st.write(f"- **Area**: {area}")
if item is not None:
    st.write(f"- **Item**: {item}")
st.write(f"- **Year**: {year}")
st.write(f"- **Rainfall**: {rainfall} mm")
st.write(f"- **Pesticides**: {pesticides} tonnes")
st.write(f"- **Temperature**: {temp} °C")


if mode == "🔮 Predict a yield":
    if st.button("🚀 Predict Yield", type="primary"):
        payload = {
            "Area": area,
            "Item": item,
            "Year": year,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temp,
        }

        with st.spinner("Calling the FastAPI model server..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict", json=payload, timeout=API_TIMEOUT_SECONDS
                )
                response.raise_for_status()

                pred = response.json()["prediction"]
                unit = response.json()["unit"]

                st.success(f"### Predicted Yield: {pred:,.2f} {unit}")
                st.balloons()

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Failed to connect to the API at {API_URL}. Is it running?"
                )
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

else:
    if st.button("🏆 Recommend Best Crop", type="primary"):
        payload = {
            "Area": area,
            "Year": year,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temp,
        }

        with st.spinner("Simulating yield for every crop via the API..."):
            try:
                response = requests.post(
                    f"{API_URL}/recommend", json=payload, timeout=API_TIMEOUT_SECONDS
                )
                response.raise_for_status()

                data = response.json()
                unit = data["unit"]
                ranking = pd.DataFrame(data["recommendations"])

                best = ranking.iloc[0]
                st.success(
                    f"### 🥇 Best Crop: {best['Item']} "
                    f"({best['relative_score']:,.2f}x its usual yield, "
                    f"{best['prediction']:,.2f} {unit})"
                )
                st.caption(
                    "Ranked by relative score = predicted yield ÷ this crop's own "
                    "global reference yield, so crops that are naturally high-yield "
                    "(e.g. potatoes) don't automatically win regardless of climate."
                )
                st.balloons()

                st.write("#### Relative Score by Crop")
                st.bar_chart(ranking.set_index("Item")["relative_score"])

                st.write("#### Ranking")
                table = ranking.rename(
                    columns={
                        "Item": "Crop",
                        "prediction": f"Predicted Yield ({unit})",
                        "relative_score": "Relative Score",
                    }
                )
                table.index = range(1, len(table) + 1)
                table.index.name = "Rank"
                st.dataframe(table, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Failed to connect to the API at {API_URL}. Is it running?"
                )
            except Exception as e:
                st.error(f"❌ Error during recommendation: {e}")
