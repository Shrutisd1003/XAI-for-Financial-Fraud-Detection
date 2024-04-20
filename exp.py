import pickle
import pandas as pd
from Modules.llm_explainer import generate_response

df = pd.read_csv("modified_data.csv")

with open('Model building/skf_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

idx = 0
response = generate_response(model, df, idx, 0)

print(response)