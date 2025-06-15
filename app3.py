import streamlit as st
import pandas as pd
import numpy as np
import joblib
import difflib
import plotly.express as px

# Load the trained model pipeline
model = joblib.load("model_curated.pkl")
feature_cols = joblib.load("features_cols.pkl")


st.set_page_config(page_title="ClamaSense", layout="wide")

st.title("ClamaSense")
st.markdown("Upload your climate data or manually input values to forecast potential extreme weather outcomes.")

# ------------------- File Upload -------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Soft column matcher
def match_columns(uploaded_df, model_features):
    matched = {}
    for feat in model_features:
        candidates = difflib.get_close_matches(feat.lower(), [col.lower() for col in uploaded_df.columns], n=1, cutoff=0.7)
        if candidates:
            matched[feat] = [col for col in uploaded_df.columns if col.lower() == candidates[0]][0]
    return matched

# ------------------- Make Predictions -------------------
def predict_on_data(df):
    # Match uploaded columns with model feature columns
    mapping = match_columns(df, feature_cols)
    missing = [col for col in feature_cols if col not in mapping]
    
    if missing:
        st.warning(f"❗ Missing required features for prediction: {missing}")
        return None

    # Reorder & rename columns to match model input
    input_df = df.rename(columns=mapping)[list(mapping.values())]
    preds = model.predict(input_df)
    df["Prediction"] = preds
    return df

# ------------------- File Upload Prediction -------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        prediction_df = predict_on_data(df)

        if prediction_df is not None:
            st.subheader("📈 Predictions")
            st.dataframe(prediction_df)

            # 📥 Download button
            csv = prediction_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")

            # 🌍 Country-based visualization
            if "Country" in prediction_df.columns:
                st.subheader("🌍 Average Prediction by Country")
                fig = px.bar(prediction_df.groupby("Country")["Prediction"].mean().reset_index(), x="Country", y="Prediction")
                st.plotly_chart(fig, use_container_width=True)

            # 📅 Year-based visualization
            if "Year" in prediction_df.columns:
                st.subheader("📅 Average Prediction by Year")
                fig = px.line(prediction_df.groupby("Year")["Prediction"].mean().reset_index(), x="Year", y="Prediction")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ Could not generate predictions due to missing features.")

    except Exception as e:
        st.error(f"Error reading file: {e}")

# ------------------- Manual Input -------------------
st.sidebar.markdown("---")
st.sidebar.header("✍️ Manual Input")

if st.sidebar.checkbox("Use manual input"):
    user_input = {}
    st.subheader("🔍 Predict for Custom Weather Data")

    for col in feature_cols:
        if col.lower() == "country":
            user_input[col] = st.text_input(f"{col} (e.g., Nigeria)")
        elif col.lower() == "year":
            user_input[col] = st.number_input(f"{col}", min_value=1900, max_value=2100, step=1)
        else:
            user_input[col] = st.number_input(f"{col}", format="%.2f")

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_input])
            pred = model.predict(input_df)[0]
            st.success(f"🌦️ Predicted Extreme Weather Score: **{pred:.2f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
