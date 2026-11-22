import streamlit as st
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("concrete_strength_model.joblib")
FEATURES = [
    ("Cement", "kg/m3", 278.6),
    ("Blast Furnace Slag", "kg/m3", 72.0),
    ("Fly Ash", "kg/m3", 55.5),
    ("Water", "kg/m3", 182.1),
    ("Superplasticizer", "kg/m3", 6.0),
    ("Coarse Aggregate", "kg/m3", 974.4),
    ("Fine Aggregate", "kg/m3", 772.7),
    ("Age", "days", 45.9),
]
TRAIN_RANGES = {
    "Cement": (102.0, 540.0), "Blast Furnace Slag": (0.0, 359.4),
    "Fly Ash": (0.0, 200.1), "Water": (121.75, 247.0),
    "Superplasticizer": (0.0, 32.2), "Coarse Aggregate": (801.0, 1145.0),
    "Fine Aggregate": (594.0, 992.6), "Age": (1.0, 365.0),
}

st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🪨", layout="centered")
st.title("Concrete Compressive Strength Predictor")
st.caption("Educational prototype(ML assgnment#1) | HistGradientBoosting regression model | Output in MPa")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

with st.form("mix_inputs"):
    values = []
    for name, unit, default in FEATURES:
        values.append(st.number_input(f"{name} ({unit})", min_value=0.0, value=float(default), step=1.0, format="%.3f"))
    submitted = st.form_submit_button("Predict compressive strength", type="primary")

if submitted:
    if values[7] <= 0:
        st.error("Age must be greater than zero days.")
    else:
        x = np.asarray(values, dtype=float).reshape(1, -1)
        prediction = float(model.predict(x)[0])
        st.metric("Predicted compressive strength", f"{prediction:.2f} MPa")
        outside = []
        for (name, _, _), val in zip(FEATURES, values):
            lo, hi = TRAIN_RANGES[name]
            if val < lo or val > hi:
                outside.append(f"{name}: {val:g} outside observed range [{lo:g}, {hi:g}]")
        if outside:
            st.warning("One or more inputs are outside the observed training-data range. This may be unreliable.\n\n" + "\n".join(outside))
        else:
            st.info("All inputs fall within the observed minimum-to-maximum ranges of the dataset. ")

st.divider()
st.caption("Model: HistGradientBoostingRegressor | Training data: 854 rows after model selection | Test RMSE: 3.92 MPa | Test R2: 0.935")
