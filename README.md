# XAI-for-Financial-Fraud-Detection
This project focuses on using eXplainable Artificial Intelligence (XAI) techniques for detecting financial fraud. The aim is to not only build accurate models for fraud detection but also to provide insights into how these models arrive at their decisions, thus enhancing transparency and trust.

## Dataset Description
The dataset used for this project contains transactions with the following parameters:
- Transaction Type: Specifies the nature of the transaction, including CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER.
- Amount: Denotes the monetary value of the transaction in the local currency.
- nameOrig: Identifies the customer who initiated the transaction.
- oldbalanceOrg: Represents the initial account balance before the transaction.
- newbalanceOrig: Reflects the updated account balance after the transaction.
- nameDest: Indicates the recipient of the transaction.
- oldbalanceDest: Displays the initial balance of the recipient before the transaction.
- newbalanceDest: Shows the new balance of the recipient after the transaction.
- unusuallogin: Quantifies the number of unusual logins associated with the transaction.
- accage: Represents the age of the account in years.
- Acct type: Specifies the type of account, either current or savings.
- Date of Transaction: Records the date when the transaction occurred.
- Time of Day: Categorizes the time of the transaction as morning, afternoon, or night.
- isFraud: Binary flag indicating whether the transaction is fraudulent or not. It takes the value 0 for non-fraudulent transactions and 1 for fraudulent transactions.


## Project Workflow
1. Exploratory Data Analysis (EDA) and Data Cleaning:
- Removed duplicates and null values.
- Extracted date from the data column formatted as dd/mm/yyyy.
- Conducted visualizations to understand dataset trends.
- Label encoded all columns with data type 'object'.
- Printed and analyzed the correlation matrix, removing unnecessary columns.
2. Handling Imbalanced Data:
- Since the data was highly imbalanced with only 0.67% fraud instances, Synthetic Minority Over-sampling Technique (SMOTE) was applied.
3. Model Training and Comparison:
- Trained multiple models including Random Forest (with and without stratified k-fold), XGBoost (with and without stratified k-fold), and a neural network (NN) model.
- Finalized the XGBoost model and pickled it for deployment.
4. Model Deployment:
- Deployed the application on Streamlit.
- Users can input transaction details, and the output includes prediction, explanation, and a SHAP graph.
- Explanation is generated using SHAP values, which are then interpreted using a Large Language Model (Google Gemini) to provide understandable explanations to users.

## Conclusion
This project demonstrates the application of XAI techniques in the domain of financial fraud detection, aiming not only for accurate predictions but also for providing interpretable insights into model decisions. The deployed application allows users to interactively explore and understand the fraud detection process, thereby enhancing transparency and trust in the system.