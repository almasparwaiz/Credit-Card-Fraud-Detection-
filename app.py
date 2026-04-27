# app.py
# Run with:
# streamlit run app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ==========================================================
# MODEL FILES (NO BASE_DIR)
# Keep .joblib files in same folder as app.py
# ==========================================================
MODEL_PATH = "stacking_fraud_detection_model.joblib"
SCALER_PATH = "fraud_detection_scaler.joblib"

# ==========================================================
# FEATURE DEFINITIONS
# ==========================================================
BASE_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

MODEL_FEATURES = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount",
    "Amount_V1_Interaction",
    "Amount_V2_Interaction",
    "Amount_V3_Interaction",
    "Amount_V4_Interaction",
    "Amount_squared",
    "Amount_cubed",
    "V1_div_V2"
]

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler

    except Exception as e:
        st.error("❌ Model files not found.")
        st.error(str(e))
        st.stop()

model, scaler = load_artifacts()

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def feature_engineering(df):
    df = df.copy()

    df["Amount_V1_Interaction"] = df["Amount"] * df["V1"]
    df["Amount_V2_Interaction"] = df["Amount"] * df["V2"]
    df["Amount_V3_Interaction"] = df["Amount"] * df["V3"]
    df["Amount_V4_Interaction"] = df["Amount"] * df["V4"]

    df["Amount_squared"] = df["Amount"] ** 2
    df["Amount_cubed"] = df["Amount"] ** 3
    df["V1_div_V2"] = df["V1"] / (df["V2"] + 1e-6)

    return df

# ==========================================================
# HEADER
# ==========================================================
st.title("💳 Credit Card Fraud Detection System")

st.write(
    "Detect suspicious transactions instantly using an advanced "
    "Stacking Classifier model."
)

# ==========================================================
# INPUT METHOD
# ==========================================================
st.markdown("---")

input_method = st.radio(
    "Choose Input Method",
    ["Upload CSV File", "Manual Entry"],
    horizontal=True
)

processed_scaled = None
raw_data = None

# ==========================================================
# CSV INPUT
# ==========================================================
if input_method == "Upload CSV File":

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:

        try:
            raw_data = pd.read_csv(uploaded_file)

            st.subheader("Preview")
            st.dataframe(raw_data.head(), use_container_width=True)

            engineered = feature_engineering(raw_data)

            X = engineered[MODEL_FEATURES]
            scaled = scaler.transform(X)

            processed_scaled = pd.DataFrame(
                scaled,
                columns=MODEL_FEATURES
            )

            st.success("✅ File ready for prediction.")

        except Exception as e:
            st.error(str(e))

# ==========================================================
# MANUAL INPUT
# ==========================================================
else:

    inputs = {}
    cols = st.columns(4)

    for i, feature in enumerate(BASE_FEATURES):

        with cols[i % 4]:

            if feature == "Amount":
                inputs[feature] = st.number_input(
                    feature,
                    value=100.0
                )
            else:
                inputs[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f"
                )

    raw_data = pd.DataFrame([inputs])

    engineered = feature_engineering(raw_data)

    X = engineered[MODEL_FEATURES]
    scaled = scaler.transform(X)

    processed_scaled = pd.DataFrame(
        scaled,
        columns=MODEL_FEATURES
    )

# ==========================================================
# PREDICTION
# ==========================================================
st.markdown("---")

if processed_scaled is not None:

    if st.button("🔍 Predict Fraud", use_container_width=True):

        try:
            predictions = model.predict(processed_scaled)
            probabilities = model.predict_proba(processed_scaled)[:, 1]

            fraud_percent = probabilities * 100

            st.header("📊 Prediction Results")

            # CSV MODE
            if input_method == "Upload CSV File":

                result_df = raw_data.copy()
                result_df["Predicted_Class"] = predictions
                result_df["Fraud_Probability (%)"] = fraud_percent.round(2)

                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇ Download Results CSV",
                    csv,
                    "fraud_results.csv",
                    "text/csv",
                    use_container_width=True
                )

            # MANUAL MODE
            else:

                pred = predictions[0]
                prob = fraud_percent[0]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Prediction",
                        "Fraudulent" if pred == 1 else "Legitimate"
                    )

                with col2:
                    st.metric(
                        "Fraud Probability",
                        f"{prob:.2f}%"
                    )

                if pred == 1:
                    st.error("🚨 High Risk Transaction")
                else:
                    st.success("✅ Legitimate Transaction")

        except Exception as e:
            st.error(str(e))

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
