import os
import shap
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

def generate_response(model, data, idx, prediction):
    input_data = data.iloc[idx,:]
    transaction_type, branch, amount, origin_old_balance, origin_new_balance, destination_old_balance, destination_new_balance, unusualLogin, accAge, acc_type, timeofday = input_data
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
        - SHAP Values Explanation (explain positive and negative SHAP features and their contribution in this case)
        - Conclusion
    """
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
                    "shap_values": shap_values,
                    "prediction": prediction
                })
    
    return response