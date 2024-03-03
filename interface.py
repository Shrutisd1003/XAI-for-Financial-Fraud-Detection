import streamlit as st
from streamlit_shap import st_shap
import numpy as np
from xgboost import XGBClassifier
import shap
import streamlit.components.v1 as components
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from dotenv import load_dotenv
import os

load_dotenv()

model = XGBClassifier()
model.load_model('xgb_model.json')

type_encoding = {'Cash In': 0, 'Cash Out': 1, 'Debit': 2, 'Payment': 3, 'Transfer': 4}

st.title("Explainable AI for Financial Fraud Detection")
st.write("Please provide the following details for analysis:")

transaction_type = st.selectbox("Transaction Type", ["Payment", "Transfer", "Cash Out", "Debit", "Cash In"])
amount = st.number_input("Amount", format="%.2f")
origin_old_balance = st.number_input("Origin's Old Balance", format="%.2f")
destination_new_balance = st.number_input("Destination's New Balance", format="%.2f")

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

vertexai.init(project=PROJECT_ID, location=LOCATION)

llm = GenerativeModel("gemini-1.0-pro")

if st.button("Detect Fraud"):
    # st.write("Transaction Type:", transaction_type)
    # st.write("Amount:", amount)
    # st.write("Origin's Old Balance:", origin_old_balance)
    # st.write("Destination's New Balance:", destination_new_balance)

    # Prepare input data for the model
    input_data = np.array([[type_encoding.get(transaction_type), amount, origin_old_balance, destination_new_balance]])

    if input_data is not None:
        # Make prediction using the LSTM model
        prediction = model.predict(input_data)

        # st.write("Fraud Detection Result:")
        if prediction > 0.5:
            st.write("Fraudulent transaction detected!")
        else:
            st.write("No fraudulent activity detected.")

    explainer = shap.TreeExplainer(model)
    explanation = explainer(input_data)
    shap_values = explanation.values

    st.write(shap_values)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], input_data))
    st_shap(shap.plots.waterfall(explanation[0]), height=300)

    prompt = f"""
        you are an result-interpreter for a model that detects credit fruad detection.
        the model takes 4 inputs transaction type, amount, origin old balance and destination new balance and returns 0 if not fraud else 1.
        the model was given the following input : transaction type - {transaction_type}, amount - {amount}, origin old balance - {origin_old_balance}, destination new balance - {destination_new_balance} 
        and gave the following output : {prediction}.
        the SHAP values for all the features for the prediction was as follows : {shap_values} and the base value as : {explanation.base_values}.
        Your job is to explain a person with non technical background the results of the model in simple language 
    """

    responses = llm.generate_content(prompt, stream=True)
    for response in responses:
        st.write(response.text)