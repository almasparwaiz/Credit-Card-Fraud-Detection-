# app.py
# Run with:
# streamlit run "D:\Credit Card Fraud Detection app\backend\app.py"

from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ==========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# PROJECT DIRECTORY CONFIGURATION
# ==========================================================
BASE_DIR = Path(r"D:\Credit Card Fraud Detection app\backend")

MODEL_PATH = BASE_DIR / "stacking_fraud_detection_model.joblib"
SCALER_PATH = BASE_DIR / "fraud_detection_scaler.joblib"

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
# LOAD MODEL & SCALER
# ==========================================================
@st.cache_resource
def load_artifacts():
    """
    Load trained model and scaler only once.
    """
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler

    except Exception as error:
        st.error("❌ Unable to load model files.")
        st.error(str(error))
        st.stop()


model, scaler = load_artifacts()

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate engineered features exactly like training pipeline.
    """
    df = df.copy()

    missing_cols = [col for col in BASE_FEATURES if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_cols)}"
        )

    df["Amount_V1_Interaction"] = df["Amount"] * df["V1"]
    df["Amount_V2_Interaction"] = df["Amount"] * df["V2"]
    df["Amount_V3_Interaction"] = df["Amount"] * df["V3"]
    df["Amount_V4_Interaction"] = df["Amount"] * df["V4"]

    df["Amount_squared"] = df["Amount"] ** 2
    df["Amount_cubed"] = df["Amount"] ** 3

    # Avoid divide by zero
    df["V1_div_V2"] = df["V1"] / (df["V2"] + 1e-6)

    return df


# ==========================================================
# HEADER
# ==========================================================
st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    """
Detect suspicious transactions instantly using a high-performance
**Stacking Classifier Machine Learning Model** built for fraud analytics.
"""
)

# ==========================================================
# INPUT METHOD
# ==========================================================
st.markdown("---")
st.header("📥 Input Transaction Data")

input_method = st.radio(
    "Choose Input Method",
    ["Upload CSV File", "Manual Entry"],
    horizontal=True
)

processed_scaled = None
raw_data = None

# ==========================================================
# CSV MODE
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

            st.success("✅ File processed successfully.")

        except Exception as error:
            st.error("❌ Error processing file.")
            st.error(str(error))

# ==========================================================
# MANUAL ENTRY MODE
# ==========================================================
else:
    st.subheader("Enter Transaction Details")

    inputs = {}
    cols = st.columns(4)

    for idx, feature in enumerate(BASE_FEATURES):
        with cols[idx % 4]:
            if feature == "Amount":
                inputs[feature] = st.number_input(
                    feature,
                    value=100.0,
                    step=1.0,
                    format="%.2f"
                )
            else:
                inputs[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f"
                )

    raw_data = pd.DataFrame([inputs])

    try:
        engineered = feature_engineering(raw_data)

        X = engineered[MODEL_FEATURES]
        scaled = scaler.transform(X)

        processed_scaled = pd.DataFrame(
            scaled,
            columns=MODEL_FEATURES
        )

        st.success("✅ Manual transaction ready for prediction.")

    except Exception as error:
        st.error(str(error))

# ==========================================================
# PREDICTION SECTION
# ==========================================================
st.markdown("---")

if processed_scaled is not None:

    if st.button("🔍 Predict Fraud", use_container_width=True):

        try:
            predictions = model.predict(processed_scaled)
            probabilities = model.predict_proba(processed_scaled)[:, 1]

            st.header("📊 Prediction Results")

            # -----------------------------------------
            # BULK CSV MODE
            # -----------------------------------------
            if input_method == "Upload CSV File":

                result_df = raw_data.copy()
                result_df["Predicted_Class"] = predictions
                result_df["Fraud_Probability"] = probabilities.round(4)

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                fraud_cases = int(result_df["Predicted_Class"].sum())
                total_rows = len(result_df)

                st.warning(
                    f"⚠️ Detected {fraud_cases} suspicious transactions out of {total_rows}"
                )

                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇ Download Results CSV",
                    data=csv,
                    file_name="fraud_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # -----------------------------------------
            # SINGLE ENTRY MODE
            # -----------------------------------------
            else:

                pred = predictions[0]
                prob = probabilities[0]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Prediction",
                        "Fraudulent" if pred == 1 else "Legitimate"
                    )

                with col2:
                    st.metric(
                        "Fraud Probability",
                        f"{prob:.4f}"
                    )

                if pred == 1:
                    st.error("🚨 High Risk Transaction Detected")
                else:
                    st.success("✅ Legitimate Transaction")

        except Exception as error:
            st.error("Prediction failed.")
            st.error(str(error))

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Professional ML Deployment")