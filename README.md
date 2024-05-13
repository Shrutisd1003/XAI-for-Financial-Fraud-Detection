# XAI-for-Financial-Fraud-Detection
This project focuses on using Explainable Artificial Intelligence (XAI) techniques for detecting financial fraud. The aim is to not only build accurate models for fraud detection but also to provide insights into how these models arrive at their decisions, thus enhancing transparency and trust.

## Dataset Description
The dataset used for this project contains transactions with the following parameters:
- Transaction ID: A unique identifier assigned to each transaction.
- Branch: The specific branch or location where the transaction occurred.
- Amount: The monetary value of the transaction in the local currency.
- Sender: The account ID of the sender initiating the transaction.
- Sender Old Balance: The balance of the sender's account before the transaction.
- Sender New Balance: The balance of the sender's account after the transaction.
- Receiver: The account ID of the recipient receiving the transaction.
- Receiver Old Balance: The balance of the receiver's account before the transaction.
- Receiver New Balance: The balance of the receiver's account after the transaction.
- Transaction Count: Number of transactions made by the sender or receiver.
- Account Age: Age of the sender or receiver's account in days.
- Last Transaction: Time since the sender or receiver's last transaction in days.
- Time of Day: Categorization of the time of the transaction into morning, afternoon, or night.
- isFraud: A binary flag indicating whether the transaction is fraudulent (1) or not (0).
About 20% of the transactions in the dataset were identified as fraudulent.

## Project Workflow
1. Exploratory Data Analysis (EDA) and Data Cleaning:
- Removed duplicates and null values.
- Conducted visualizations to understand dataset trends.
- Label encoded all columns with data type 'object'.
- Printed and analyzed the correlation matrix, removing unnecessary columns.
2. Model Training and Comparison:
- Trained multiple models including Logistic Regressor, Random Forest, Decision Tree and Naive Bayes.
- Evaluated the models using stratified k-fold cross-validation with 25 splits.
  <img width="609" alt="image" src="https://github.com/Shrutisd1003/XAI-for-Financial-Fraud-Detection/assets/113239067/4a34d183-22a5-4d88-80dd-cd5005901801">
- Selected the Random Forest model as the final choice and optimized its hyperparameters.
- Saved the final model using pickle for future use.
3. Model Deployment:
- Implemented deployment of the application on Streamlit for convenient usage by the bank.
- Enabled the bank to upload datasets for making predictions.
- Provided the dataset with fraudulent transactions highlighted in red for easy identification.
- Offered various customizable visualizations for enhanced analysis.
- Explanation is generated using SHAP values, which are then interpreted using a Large Language Model (Google Gemini) to provide understandable explanations to users.

## Conclusion
This project demonstrates the application of XAI techniques in the domain of financial fraud detection, aiming not only for accurate predictions but also for providing interpretable insights into model decisions. The deployed application allows banks to interactively explore and understand the fraud detection process, thereby enhancing transparency and trust in the system.
