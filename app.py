import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from Modules.data_cleaning import cleaned_data
from Modules.llm_explainer import generate_response

def load_model():
    with open('Model building/prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_and_clean_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data.to_csv("original_data.csv", index=False)
    transaction_ids, modified_data = cleaned_data(data)
    return transaction_ids, modified_data

def color_coding(row):
    return ['background-color: red' if row["Prediction"] == 'Fraud' else '' for _ in row]

def stream_data(words):
    for word in words.split(" "):
        yield word + " "
        time.sleep(0.02)

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        transaction_ids, modified_data = load_and_clean_data(uploaded_file)
        # st.write(modified_data)
        model = load_model()
        prediction = model.predict(modified_data).round().astype(int)
        
        original_data = pd.read_csv("original_data.csv")
        original_data["Prediction"] = ["Fraud" if x == 1 else "Not Fraud" for x in prediction]
        st.dataframe(original_data.style.apply(color_coding, axis=1), hide_index=True)
        
        fraud_count = np.sum(prediction)
        not_fraud_count = len(prediction) - fraud_count
        colors = ['#ff9999', '#66b3ff']
        labels=['Fraud', 'Not Fraud']
        fig1, ax1 = plt.subplots(figsize=(9,3))
        ax1.pie([fraud_count, not_fraud_count], colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        transaction_ids = transaction_ids.tolist()
        transaction_ids.insert(0, "-Select-")
        transaction_id = st.selectbox("Transaction IDs", transaction_ids)

        if transaction_id != "-Select-":
            idx = transaction_ids.index(transaction_id) - 1
            if st.button("Explain"):
                try:
                    with st.spinner('Analyzing...'):
                        response = generate_response(model, modified_data, idx, prediction[idx])
                        st.write(response)
                        # st.write_stream(stream_data(response))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.write("Select a file")

if __name__ == "__main__":
    main()
