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
model.load_model('xgb_final.json')

type_encoding = {'Cash In': 0, 'Cash Out': 1, 'Debit': 2, 'Payment': 3, 'Transfer': 4}
acctype_encoding = {'Current': 0, 'Savings': 1}
timeofday_encoding = {'Afternoon': 0, 'Morning': 1, 'Night': 2}
branch_encoding = {'Afganistan': 0, 'Albania': 1, 'Alemania': 2, 'Angola': 3, 'Arabia Saudi': 4, 'Argelia': 5, 'Argentina': 6, 
                    'Australia': 7, 'Austria': 8, 'Azerbaiyan': 9, 'Banglades': 10, 'Barbados': 11, 'Barein': 12, 'Belgica': 13,
                    'Benin': 14, 'Bielorrusia': 15, 'Bolivia': 16, 'Bosnia y Herzegovina': 17, 'Brasil': 18, 'Bulgaria': 19, 
                    'Burkina Faso': 20, 'Camboya': 21, 'Camerun': 22, 'Canada': 23, 'Chile': 24, 'China': 25, 'Colombia': 26, 
                    'Corea del Sur': 27, 'Costa Rica': 28, 'Costa de Marfil': 29, 'Croacia': 30, 'Cuba': 31, 'Dinamarca': 32, 
                    'Ecuador': 33, 'Egipto': 34, 'El Salvador': 35, 'Emiratos arabes Unidos': 36, 'Eritrea': 37, 'Eslovaquia': 38, 
                    'Espana': 39, 'Estados Unidos': 40, 'Etiopia': 41, 'Filipinas': 42, 'Finlandia': 43, 'Francia': 44, 
                    'Georgia': 45, 'Ghana': 46, 'Guadalupe': 47, 'Guatemala': 48, 'Guinea': 49, 'Haiti': 50, 'Honduras': 51, 
                    'Hong Kong': 52, 'Hungria': 53, 'India': 54, 'Indonesia': 55, 'Irak': 56, 'Iran': 57, 'Irlanda': 58, 
                    'Israel': 59, 'Italia': 60, 'Jamaica': 61, 'Japon': 62, 'Jordania': 63, 'Kazajistan': 64, 'Kenia': 65, 
                    'Kirguistan': 66, 'Laos': 67, 'Lesoto': 68, 'Liberia': 69, 'Libia': 70, 'Lituania': 71, 'Macedonia': 72, 
                    'Madagascar': 73, 'Malasia': 74, 'Mali': 75, 'Marruecos': 76, 'Martinica': 77, 'Mexico': 78, 'Moldavia': 79, 
                    'Mongolia': 80, 'Mozambique': 81, 'Myanmar (Birmania)': 82, 'Nepal': 83, 'Nicaragua': 84, 'Niger': 85, 
                    'Nigeria': 86, 'Noruega': 87, 'Nueva Zelanda': 88, 'Paises Bajos': 89, 'Pakistan': 90, 'Panama': 91, 
                    'Papua Nueva Guinea': 92, 'Paraguay': 93, 'Peru': 94, 'Polonia': 95, 'Portugal': 96, 'Qatar': 97, 
                    'Reino Unido': 98, 'Republica Checa': 99, 'Republica Democratica del Congo': 100, 'Republica Dominicana': 101, 
                    'Republica del Congo': 102, 'Ruanda': 103, 'Rumania': 104, 'Rusia': 105, 'Senegal': 106, 'Sierra Leona': 107, 
                    'Singapur': 108, 'Siria': 109, 'Somalia': 110, 'Sri Lanka': 111, 'SudAfrica': 112, 'Sudan': 113, 
                    'Suecia': 114, 'Suiza': 115, 'Tailandia': 116, 'Tanzania': 117, 'Tayikistan': 118, 'Togo': 119, 
                    'Trinidad y Tobago': 120, 'Tunez': 121, 'Turkmenistan': 122, 'Turquia': 123, 'Ucrania': 124, 'Uganda': 125, 
                    'Uruguay': 126, 'Uzbekistan': 127, 'Venezuela': 128, 'Vietnam': 129, 'Yemen': 130, 'Yibuti': 131, 
                    'Zambia': 132, 'Zimbabue': 133}

st.title("Explainable AI for Financial Fraud Detection")
st.write("Please provide the following details for analysis:")

transaction_type = st.selectbox("Transaction Type", ["Payment", "Transfer", "Cash Out", "Debit", "Cash In"])
branch = st.selectbox("Branch", ['Indonesia', 'India', 'Australia', 'China', 'Japon',
                                'Corea del Sur', 'Singapur', 'Turquia', 'Mongolia',
                                'Estados Unidos', 'Nigeria', 'Republica Democratica del Congo',
                                'Senegal', 'Marruecos', 'Alemania', 'Paises Bajos', 'Reino Unido',
                                'Francia', 'Guatemala', 'El Salvador', 'Panama',
                                'Republica Dominicana', 'Venezuela', 'Colombia', 'Honduras',
                                'Brasil', 'Mexico', 'Cuba', 'Peru', 'Nicaragua', 'Argentina',
                                'Ecuador', 'Angola', 'Sudan', 'Somalia', 'Costa de Marfil',
                                'Egipto', 'Italia', 'Espana', 'Suecia', 'Austria', 'Canada',
                                'Madagascar', 'Argelia', 'Liberia', 'Zambia', 'Niger', 'SudAfrica',
                                'Mozambique', 'Tanzania', 'Ruanda', 'Israel', 'Nueva Zelanda',
                                'Banglades', 'Tailandia', 'Irak', 'Arabia Saudi', 'Filipinas',
                                'Kazajistan', 'Iran', 'Myanmar (Birmania)', 'Uzbekistan', 'Benin',
                                'Camerun', 'Kenia', 'Togo', 'Ucrania', 'Polonia', 'Portugal',
                                'Rumania', 'Trinidad y Tobago', 'Afganistan', 'Pakistan',
                                'Vietnam', 'Malasia', 'Finlandia', 'Rusia', 'Irlanda', 'Noruega',
                                'Eslovaquia', 'Belgica', 'Bolivia', 'Chile', 'Jamaica', 'Yemen',
                                'Ghana', 'Guinea', 'Etiopia', 'Bulgaria', 'Kirguistan', 'Georgia',
                                'Nepal', 'Emiratos arabes Unidos', 'Camboya', 'Uganda', 'Lesoto',
                                'Lituania', 'Suiza', 'Hungria', 'Dinamarca', 'Haiti',
                                'Bielorrusia', 'Croacia', 'Laos', 'Barein', 'Macedonia',
                                'Republica Checa', 'Sri Lanka', 'Zimbabue', 'Eritrea',
                                'Burkina Faso', 'Costa Rica', 'Libia', 'Uruguay', 'Barbados',
                                'Tayikistan', 'Siria', 'Guadalupe', 'Papua Nueva Guinea',
                                'Azerbaiyan', 'Turkmenistan', 'Paraguay', 'Jordania', 'Hong Kong',
                                'Martinica', 'Moldavia', 'Qatar', 'Mali', 'Albania',
                                'Bosnia y Herzegovina', 'Republica del Congo', 'Tunez',
                                'Sierra Leona', 'Yibuti'])
amount = st.number_input("Amount", format="%.2f")
origin_old_balance = st.number_input("Origin's Old Balance", format="%.2f")
origin_new_balance = st.number_input("Origin's New Balance", format="%.2f")
destination_old_balance = st.number_input("Destination's Old Balance", format="%.2f")
destination_new_balance = st.number_input("Destination's New Balance", format="%.2f")
unusualLogin = st.number_input("Number of unusual logins", format="%.0f")
accAge = st.number_input("Account age", format="%.2f")
acc_type = st.selectbox("Account Type", ['Current', 'Savings'])
timeofday = st.selectbox("Time of day", ['Morning', 'Afternoon', 'Night'])
date = st.number_input("Date", format="%.0f")

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
    input_data = np.array([[type_encoding.get(transaction_type), branch_encoding.get(branch), amount, origin_old_balance, origin_new_balance, destination_old_balance, destination_new_balance, unusualLogin, accAge, acctype_encoding.get(acc_type), timeofday_encoding.get(timeofday), date]])

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