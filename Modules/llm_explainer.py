import os
import shap
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
from dotenv import load_dotenv
from Modules.load_encodings import get_encodings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

def generate_response(model, data, idx, prediction):
    input_data = data.iloc[idx,:]
    branch, amount, sender, sender_old_balance, sender_new_balance, receiver, receiver_old_balance, receiver_new_balance, transaction_count, account_age, last_transaction = input_data

    explainer = shap.TreeExplainer(model)
    explanation = explainer(input_data)
    shap_values = explanation.values

    branch_encoding, sender_encoding, receiver_encoding = get_encodings()

    template = f"""
        You are a fraud detection result interpreter, helping users understand the outcome of our credit fraud detection model in simple terms. Here's a breakdown of the analysis:

        The model evaluates transactions to determine if they are potentially fraudulent. 
        It considers several factors, including the type of transaction, the amount involved, and the balances of the sender and recipient accounts. 

        In the recent analysis, the model examined a transaction with the following details:
        - Branch: {list(branch_encoding.keys())[int(branch)]}
        - Amount: {amount}
        - Sender's Account ID: {list(sender_encoding.keys())[int(sender)]}
        - Sender's Old Balance: {sender_old_balance}
        - Sender's New Balance: {sender_new_balance}
        - Receiver's Account ID: {list(receiver_encoding.keys())[int(receiver)]}
        - Receiver's Old Balance: {receiver_old_balance}
        - Receiver's New Balance: {receiver_new_balance}
        - Number of transactions made by sender: {transaction_count}
        - Account's age: {account_age}
        - Last transaction made by user: {last_transaction}

        Based on this information, the model provided its prediction that is {prediction}. 
        If the prediction value is 0, it indicates that no fraudulent activity is detected. 
        Conversely, if the prediction value is 1, it suggests that the transaction is potentially fraudulent.

        Additionally, the model generates SHAP (SHapley Additive exPlanations) values to explain its decision-making process. 
        These values indicate the impact of each feature on the prediction. 
        A positive SHAP value suggests an increase in the likelihood of fraud, while a negative value indicates a decrease.

        Here are the SHAP values for the features considered: {shap_values}

        The base value represents the model's average prediction across all observations. 
        Understanding these SHAP values can provide insights into why the model made a particular prediction.

        Your role is to communicate these findings in a clear and understandable manner, ensuring that individuals without a technical background grasp the implications of the model's analysis.
        Format of output:
        - Fraud Detection Analysis Report
        - Transcation Details (list all the parameters and their values)
        - Model Prediction
        - SHAP Values Explanation (explain SHAP features that heavily contribute towards the decision and the inconsistencies in the sender and receiver account balances if any)
        - Conclusion
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
                    "branch": branch,
                    "amount": amount,
                    "sender": sender,
                    "sender_old_balance": sender_old_balance,
                    "sender_new_balance": sender_new_balance,
                    "receiver": receiver,
                    "receiver_old_balance": receiver_old_balance,
                    "receiver_new_balance": receiver_new_balance,
                    "transaction_count": transaction_count,
                    "account_age": account_age,
                    "last_transaction": last_transaction,
                    "shap_values": shap_values,
                    "prediction": prediction
                })
    
    return response