import os
import ast
import shap
import time
import pickle
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from xgboost import XGBClassifier
from streamlit_shap import st_shap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

with open('skf_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

def read_encoding(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readline()

def stream_data(words):
    for word in words.split(" "):
        yield word + " "
        time.sleep(0.02)

encodings = {}

encoding_files = ['type_encoding.txt', 'acctype_encoding.txt', 'timeofday_encoding.txt', 'branch_encoding.txt']

for file_name in encoding_files:
    key = file_name.split('_')[0] + '_encoding'
    encodings[key] = read_encoding(f'Encodings/{file_name}')

type_encoding = ast.literal_eval(encodings['type_encoding'])
acctype_encoding = ast.literal_eval(encodings['acctype_encoding'])
timeofday_encoding = ast.literal_eval(encodings['timeofday_encoding'])
branch_encoding = ast.literal_eval(encodings['branch_encoding'])

st.title("XAI for Financial Fraud Detection")
st.write("Please provide the following details for analysis:")

transaction_type = st.selectbox("Transaction Type", ["-Select-", "Payment", "Transfer", "Cash Out", "Debit", "Cash In"])
branch_names = list(branch_encoding.keys())
branch_names.insert(0, "-Select-")
branch = st.selectbox("Branch", branch_names)
amount = st.number_input("Amount", format="%.2f")
origin_old_balance = st.number_input("Origin's Old Balance", format="%.2f")
origin_new_balance = st.number_input("Origin's New Balance", format="%.2f")
destination_old_balance = st.number_input("Destination's Old Balance", format="%.2f")
destination_new_balance = st.number_input("Destination's New Balance", format="%.2f")
unusualLogin = st.number_input("Number of unusual logins", format="%.0f")
accAge = st.number_input("Account age", format="%.2f")
acc_type = st.selectbox("Account Type", ["-Select-", 'Current', 'Savings'])
timeofday = st.selectbox("Time of day", ["-Select-", 'Morning', 'Afternoon', 'Night'])
date = st.number_input("Date", format="%.0f")

if st.button("Detect Fraud"):
    input_data = np.array([[type_encoding.get(transaction_type), branch_encoding.get(branch), amount, origin_old_balance, origin_new_balance, destination_old_balance, destination_new_balance, unusualLogin, accAge, acctype_encoding.get(acc_type), timeofday_encoding.get(timeofday), date]])

    if input_data is not None:
        prediction = model.predict(input_data)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(input_data)
    shap_values = explanation.values

    template = f"""
        You are a fraud detection result interpreter, helping users understand the outcome of our credit fraud detection model in simple terms. Here's a breakdown of the analysis:

        The model evaluates transactions to determine if they are potentially fraudulent. 
        It considers several factors, including the type of transaction, the amount involved, and the balances of the sender and recipient accounts. 

        In the recent analysis, the model examined a transaction with the following details:
        - Transaction Type: {transaction_type}
        - Branch: {branch}
        - Amount: {amount}
        - Origin's Old Balance: {origin_old_balance}
        - Origin's New Balance: {origin_new_balance}
        - Destination's Old Balance: {destination_old_balance}
        - Destination's New Balance: {destination_new_balance}
        - Number of unusual logins: {unusualLogin}
        - Account's age: {accAge}
        - Account type: {acc_type}
        - Time of day: {timeofday}
        - Date: {date}

        Based on this information, the model provided its prediction that is {prediction}. 
        If the prediction value is 0, it indicates that no fraudulent activity is detected. 
        Conversely, if the prediction value is 1, it suggests that the transaction is potentially fraudulent.

        Additionally, the model generates SHAP (SHapley Additive exPlanations) values to explain its decision-making process. 
        These values indicate the impact of each feature on the prediction. 
        A positive SHAP value suggests an increase in the likelihood of fraud, while a negative value indicates a decrease.

        Here are the SHAP values for the features considered:
        {shap_values}

        The base value represents the model's average prediction across all observations. 
        Understanding these SHAP values can provide insights into why the model made a particular prediction.

        Your role is to communicate these findings in a clear and understandable manner, ensuring that individuals without a technical background grasp the implications of the model's analysis.
        Format of output:
        - Fraud Detection Analysis Report
        - Transcation Details (list all the parameters and their values)
        - Model Prediction
        - SHAP Values Explanation (explain positive and negative SHAP features and their contribution in this case)
        - Conclusion
    """
    with st.spinner('Analyzing...'):
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
                        "transaction_type": transaction_type, 
                        "branch": branch,
                        "amount": amount,
                        "origin_old_balance": origin_old_balance,
                        "origin_new_balance": origin_new_balance,
                        "destination_old_balance": destination_old_balance,
                        "destination_new_balance": destination_new_balance,
                        "unusualLogin": unusualLogin,
                        "accAge": accAge,
                        "acc_type": acc_type,
                        "timeofday": timeofday,
                        "date": date,
                        "shap_values": shap_values,
                        "prediction": prediction
                    })

    if prediction > 0.5:
        st.write_stream(stream_data("Fraudulent transaction detected!"))
    else:
        st.write_stream(stream_data("No fraudulent activity detected."))
    st.write_stream(stream_data(response))
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], input_data))