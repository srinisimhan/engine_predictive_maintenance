# Importing packages
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import io

# App Configuration
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Smart Engine Predictive Maintenance App")
st.markdown("""
This application predicts whether an engine is **Faulty (maintenance required)** or **Normal**
based on sensor readings.

**Target:**
- **0 = Normal**
- **1 = Faulty**

**Note:** The model expects engineered features, so the app computes the same feature engineering
used during training to ensure schema consistency.
""")

# Model Settings (Hugging Face)
MODEL_REPO_ID = "simnid/predictive-maintenance-model"
MODEL_FILENAME = "best_predictive_maintenance_model.joblib"

# Dataset repo (for pulling bulk sample)
DATA_REPO_ID = "simnid/predictive-engine-maintenance-dataset"
BULK_TEST_FILENAME = "bulk_test_sample.csv"

RAW_COLS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]

ENGINEERED_COLS = [
    "RPM_FuelPressure_Ratio",
    "Power_Index",
    "Thermal_Pressure_Index",
    "Mech_Cooling_Balance",
    "Pressure_Coordination",
    "Low_Oil_Pressure_Flag",
    "High_Coolant_Temp_Flag",
    "Low_RPM_Flag"
]

FINAL_FEATURE_ORDER = RAW_COLS + ENGINEERED_COLS

# Feature Engineering
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required raw columns exist
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert to numeric (safe conversion)
    for c in RAW_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[RAW_COLS].isnull().any().any():
        bad_cols = df[RAW_COLS].columns[df[RAW_COLS].isnull().any()].tolist()
        raise ValueError(f"Non-numeric / missing values detected in: {bad_cols}")

    # Interaction Features
    df["RPM_FuelPressure_Ratio"] = df["Engine rpm"] / (df["Fuel pressure"] + 1e-5)
    df["Power_Index"] = (df["Engine rpm"] * df["Fuel pressure"]) / 1000

    # System Stress Indicators
    df["Thermal_Pressure_Index"] = df["Coolant temp"] / (df["Fuel pressure"] + 1e-5)
    df["Mech_Cooling_Balance"] = (
        (df["Engine rpm"] + df["Lub oil pressure"]) -
        (df["Coolant temp"] + df["Coolant pressure"])
    )
    df["Pressure_Coordination"] = df["Fuel pressure"] - df["Coolant pressure"]

    # Early Warning Flags (data-driven thresholds)
    df["Low_Oil_Pressure_Flag"] = (df["Lub oil pressure"] < 1.5).astype(int)
    df["High_Coolant_Temp_Flag"] = (df["Coolant temp"] > 100).astype(int)
    df["Low_RPM_Flag"] = (df["Engine rpm"] < 600).astype(int)

    return df[FINAL_FEATURE_ORDER]

# Load Model
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None

model = load_model()
if model is None:
    st.warning("Model could not be loaded. Please verify model repo + filename.")
    st.stop()


# Sidebar: Business + Model Context
with st.sidebar:
    st.header("About This Model")
    st.markdown("""
**Model Details**
- **Model Type:** Gradient Boosting Classifier
- **Optimization Objective:** Maximize recall for faulty engines (minimize missed failures)
- **Artifact Source:** Hugging Face Model Hub

**Why Recall Matters**
A false negative means a failure was missed, leading to downtime, safety risks, and costly repairs.
""")

    st.subheader("Production Metrics (Reference)")
    st.metric("Recall (Faulty)", "0.84")
    st.metric("ROC-AUC", "0.70")
    st.metric("PR-AUC", "0.80")

    st.markdown("---")
    st.subheader("Decision Threshold")
    threshold = st.slider(
        "Classification Threshold (Faulty if P ≥ threshold)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01
    )
    st.caption("Lower threshold → higher recall (fewer missed failures), but more false alarms.")


# Tabs: Single + Bulk Prediction
tab1, tab2 = st.tabs(["🔎 Single Prediction", "📦 Bulk Prediction"])


# Single Prediction
with tab1:
    st.subheader("Engine Sensor Inputs")

    c1, c2, c3 = st.columns(3)

    with c1:
        engine_rpm = st.number_input("Engine rpm", min_value=0.0, value=700.0, step=1.0)
        lub_oil_pressure = st.number_input("Lub oil pressure", min_value=0.0, value=2.50, step=0.01)

    with c2:
        fuel_pressure = st.number_input("Fuel pressure", min_value=0.0, value=12.00, step=0.01)
        coolant_pressure = st.number_input("Coolant pressure", min_value=0.0, value=2.50, step=0.01)

    with c3:
        lub_oil_temp = st.number_input("lub oil temp", min_value=0.0, value=80.0, step=0.1)
        coolant_temp = st.number_input("Coolant temp", min_value=0.0, value=85.0, step=0.1)

    raw_input_df = pd.DataFrame([{
        "Engine rpm": engine_rpm,
        "Lub oil pressure": lub_oil_pressure,
        "Fuel pressure": fuel_pressure,
        "Coolant pressure": coolant_pressure,
        "lub oil temp": lub_oil_temp,
        "Coolant temp": coolant_temp
    }])

    try:
        feature_df = add_engineered_features(raw_input_df)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        st.stop()

    with st.expander("View engineered input dataframe"):
        st.dataframe(feature_df)
        csv = feature_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Engineered Input CSV", csv, "engine_input_features.csv", "text/csv")

    st.subheader("Prediction Output")

    if st.button("Predict Engine Condition", type="primary", use_container_width=True):
        try:
            proba_faulty = None
            if hasattr(model, "predict_proba"):
                proba_faulty = float(model.predict_proba(feature_df)[0][1])

            # Threshold-based classification (business control)
            if proba_faulty is not None:
                pred_class = int(proba_faulty >= threshold)
            else:
                pred_class = int(model.predict(feature_df)[0])

            colA, colB = st.columns(2)

            with colA:
                if pred_class == 1:
                    st.error("⚠️ Prediction: FAULTY (Maintenance Recommended)")
                else:
                    st.success("✅ Prediction: NORMAL (No Immediate Maintenance Required)")

            with colB:
                if proba_faulty is not None:
                    st.metric("Probability of Faulty (Class 1)", f"{proba_faulty*100:.1f}%")
                    st.progress(int(proba_faulty * 100))
                else:
                    st.info("Probability score unavailable (model does not support predict_proba).")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Bulk Prediction
with tab2:
    st.subheader("Bulk CSV Prediction")

    st.markdown("""
Upload a CSV containing **raw sensor columns only**:

- Engine rpm
- Lub oil pressure
- Fuel pressure
- Coolant pressure
- lub oil temp
- Coolant temp

The app will automatically engineer features and return:
- `Predicted_Class` (0/1)
- `Faulty_Probability` (if available)
""")

    # Try pulling a sample file from HF dataset repo (like tourism project pattern)
    @st.cache_resource
    def load_bulk_sample():
        try:
            path = hf_hub_download(
                repo_id=DATA_REPO_ID,
                filename=BULK_TEST_FILENAME,
                repo_type="dataset"
            )
            return pd.read_csv(path)
        except Exception:
            return None

    sample_df = load_bulk_sample()
    if sample_df is not None:
        with st.expander("Preview bulk sample from Hugging Face"):
            st.dataframe(sample_df.head())

    uploaded_file = st.file_uploader("Upload CSV for bulk prediction", type=["csv"])

    bulk_df = None
    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)
    elif sample_df is not None:
        bulk_df = sample_df.copy()

    if bulk_df is not None:
        st.markdown("✅ Bulk data loaded.")
        st.dataframe(bulk_df.head())

        if st.button("Run Bulk Prediction", use_container_width=True):
            try:
                # Ensure required columns exist
                missing = [c for c in RAW_COLS if c not in bulk_df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                    st.stop()

                bulk_features = add_engineered_features(bulk_df[RAW_COLS])

                # Predict
                preds = model.predict(bulk_features).astype(int)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(bulk_features)[:, 1]
                else:
                    probs = np.full(shape=(len(bulk_features),), fill_value=np.nan)

                # Threshold override if proba exists
                if hasattr(model, "predict_proba"):
                    preds = (probs >= threshold).astype(int)

                out = bulk_df.copy()
                out["Predicted_Class"] = preds
                out["Faulty_Probability"] = probs

                st.success("Bulk predictions completed.")
                st.dataframe(out.head(50))

                out_csv = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Bulk Predictions CSV",
                    out_csv,
                    "bulk_engine_predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Bulk prediction failed: {e}")


# Footer
st.markdown("---")
st.caption("Predictive Maintenance | Gradient Boosting + Streamlit + Hugging Face Model Hub")
