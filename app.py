import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")

    return model, features


model, selected_features = load_model()



st.title("💳 Credit Card Fraud Detection")
st.markdown(
    """
    This application uses a **Machine Learning model (XGBoost)**
    to detect potentially fraudulent credit card transactions.
    """
)

st.divider()



st.sidebar.title("About the Model")

st.sidebar.info(
    """
    **Model:** XGBoost

    **Task:** Fraud Detection

    **Selected Features:**
    V2, V3, V4, V9, V10, V11,
    V12, V14, V16, V17, Amount
    """
)


tab1, tab2 = st.tabs([
    "🔍 Single Transaction",
    "📂 CSV File Prediction"
])


with tab1:

    st.subheader("Enter Transaction Information")

    st.write(
        "Enter the values of the features below."
    )

    col1, col2 = st.columns(2)

    input_data = {}

    with col1:

        input_data["V2"] = st.number_input(
            "V2",
            value=0.0,
            format="%.6f"
        )

        input_data["V3"] = st.number_input(
            "V3",
            value=0.0,
            format="%.6f"
        )

        input_data["V4"] = st.number_input(
            "V4",
            value=0.0,
            format="%.6f"
        )

        input_data["V9"] = st.number_input(
            "V9",
            value=0.0,
            format="%.6f"
        )

        input_data["V10"] = st.number_input(
            "V10",
            value=0.0,
            format="%.6f"
        )

        input_data["V11"] = st.number_input(
            "V11",
            value=0.0,
            format="%.6f"
        )

    with col2:

        input_data["V12"] = st.number_input(
            "V12",
            value=0.0,
            format="%.6f"
        )

        input_data["V14"] = st.number_input(
            "V14",
            value=0.0,
            format="%.6f"
        )

        input_data["V16"] = st.number_input(
            "V16",
            value=0.0,
            format="%.6f"
        )

        input_data["V17"] = st.number_input(
            "V17",
            value=0.0,
            format="%.6f"
        )

        input_data["Amount"] = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            format="%.2f"
        )

    st.divider()

    if st.button(
        "🔎 Detect Fraud",
        type="primary",
        use_container_width=True
    ):

        # Create DataFrame
        input_df = pd.DataFrame(
            [input_data]
        )

        # Make sure feature order is correct
        input_df = input_df[selected_features]

        # Prediction
        prediction = model.predict(input_df)[0]

        # Probability
        probability = model.predict_proba(input_df)[0]

        fraud_probability = probability[1]
        normal_probability = probability[0]

        # Results
        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "🚨 Potential Fraudulent Transaction"
            )

            st.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%"
            )

        else:

            st.success(
                "✅ Normal Transaction"
            )

            st.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%"
            )

        # Probability chart
        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame(
            {
                "Class": [
                    "Normal",
                    "Fraud"
                ],
                "Probability": [
                    normal_probability,
                    fraud_probability
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index("Class")
        )



with tab2:

    st.subheader("Upload Transactions CSV")

    st.write(
        "Upload a CSV file containing the required features."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        # Check missing features
        missing_features = [
            feature
            for feature in selected_features
            if feature not in data.columns
        ]

        if missing_features:

            st.error(
                f"Missing features: {missing_features}"
            )

        else:

            if st.button(
                "🚀 Predict All Transactions",
                type="primary"
            ):

                X_input = data[selected_features]

                predictions = model.predict(X_input)

                probabilities = model.predict_proba(X_input)

                data["Prediction"] = predictions

                data["Fraud Probability"] = probabilities[:, 1]

                st.success(
                    "Predictions completed successfully!"
                )

                st.dataframe(
                    data,
                    use_container_width=True
                )

                # Statistics
                fraud_count = (predictions == 1).sum()
                normal_count = (predictions == 0).sum()

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Normal Transactions",
                        normal_count
                    )

                with col2:

                    st.metric(
                        "Fraudulent Transactions",
                        fraud_count
                    )