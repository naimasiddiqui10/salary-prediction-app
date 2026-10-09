"""Salary Prediction App: predicts salary from years of experience using a saved regression model."""
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH, METRICS_PATH, DATA_PATH = BASE / "model" / "model.pkl", BASE / "model" / "metrics.json", BASE / "data" / "salary_dataset.csv"
FEATURE, TARGET = "Experience Years", "Salary"

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")


@st.cache_resource
def load_model():
    """Load the serialized model (the model is NOT retrained when the app runs)."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


try:
    model, data, info = load_model(), load_data(), load_metrics()
except Exception as exc:  # missing file or scikit-learn version mismatch
    st.error(f"Could not load the saved model: {exc}")
    st.info("Re-run `notebooks/model_training.ipynb` to regenerate `model/model.pkl`, "
            "and make sure scikit-learn in `requirements.txt` matches the version used for training.")
    st.stop()

lo, hi = info["min_experience"], info["max_experience"]


def predict(years: float) -> float:
    return float(model.predict(pd.DataFrame({FEATURE: [years]}))[0])


if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------- header
st.title("💼 Salary Prediction App")
st.write("Estimate a salary from **years of experience** using a linear regression model trained on "
         f"{info['n_rows']} employee records. Enter an experience value below to get a prediction.")

tab_predict, tab_data, tab_model = st.tabs(["🔮 Predict", "📊 Dataset", "📈 Model performance"])

# ---------------------------------------------------------------- predict
with tab_predict:
    years = st.number_input("Years of experience", min_value=0.0, max_value=50.0, value=5.0, step=0.1,
                            help=f"The model was trained on experience between {lo} and {hi} years.")
    if years < lo or years > hi:
        st.warning(f"{years} years is outside the training range ({lo} to {hi} years). "
                   "The prediction is an extrapolation and may be unreliable.")

    if st.button("Predict salary", type="primary"):
        pred = predict(years)
        st.session_state.history.insert(0, {"Experience (years)": years, "Predicted salary": round(pred)})
        st.session_state.last = (years, pred)

    if "last" in st.session_state:
        y_last, p_last = st.session_state.last
        st.metric(f"Predicted salary for {y_last:g} years of experience", f"{p_last:,.0f}")
        st.caption(f"Typical error of the model is about ±{info['test_mae']:,.0f} "
                   f"(average) and ±{info['test_rmse']:,.0f} (RMSE) on unseen data. "
                   "Salary is shown in the units used in the dataset.")

        # chart: data, fitted line, and the user's prediction
        line = pd.DataFrame({FEATURE: [lo, hi]}).assign(**{TARGET: lambda d: model.predict(d[[FEATURE]])})
        point = pd.DataFrame({FEATURE: [y_last], TARGET: [p_last]})
        base = alt.Chart(data).mark_circle(opacity=0.35, size=40).encode(
            x=alt.X(f"{FEATURE}:Q", title="Experience (years)"), y=alt.Y(f"{TARGET}:Q", title="Salary"),
            tooltip=[FEATURE, TARGET])
        fit = alt.Chart(line).mark_line(color="black").encode(x=f"{FEATURE}:Q", y=f"{TARGET}:Q")
        mark = alt.Chart(point).mark_point(color="red", size=250, filled=True).encode(x=f"{FEATURE}:Q", y=f"{TARGET}:Q")
        st.altair_chart((base + fit + mark).properties(title="Your prediction (red) on the fitted line", height=320),
                        width="stretch")

    if st.session_state.history:
        st.subheader("Prediction history")
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch", hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.pop("last", None)
            st.rerun()

# ---------------------------------------------------------------- dataset
with tab_data:
    st.subheader("Dataset information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(data):,}")
    c2.metric("Experience range", f"{lo} to {hi} yrs")
    c3.metric("Average salary", f"{data[TARGET].mean():,.0f}")
    st.dataframe(data.describe().T.round(1), width="stretch")
    st.altair_chart(alt.Chart(data).mark_circle(opacity=0.5, size=50).encode(
        x=alt.X(f"{FEATURE}:Q", title="Experience (years)"), y=alt.Y(f"{TARGET}:Q", title="Salary"),
        tooltip=[FEATURE, TARGET]).properties(title="Salary vs experience", height=320), width="stretch")
    with st.expander("Show raw data"):
        st.dataframe(data, width="stretch", hide_index=True)

# ---------------------------------------------------------------- model
with tab_model:
    st.subheader("Model performance")
    st.write(f"**Algorithm:** {info['model']} | **Feature:** {info['feature']} | **Target:** {info['target']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (test)", f"{info['test_r2']:.3f}")
    m2.metric("MAE (test)", f"{info['test_mae']:,.0f}")
    m3.metric("RMSE (test)", f"{info['test_rmse']:,.0f}")
    m4.metric("CV R²", f"{info['cv_r2_mean']:.3f}")
    st.markdown(
        f"- **How to read this:** R² of {info['test_r2']:.2f} means the model explains about "
        f"{info['test_r2']*100:.0f}% of the variation in salary on unseen data.\n"
        f"- **Equation:** salary ≈ {info['intercept']:,.0f} + {info['coef']:,.0f} × years of experience "
        f"(each extra year adds about {info['coef']:,.0f}).\n"
        "- **Evaluation:** 80/20 train/test split and 5-fold shuffled cross-validation (the CV R² above).\n"
        "- **Limitation:** experience is the only input, so role, education, location and company are not captured.")
