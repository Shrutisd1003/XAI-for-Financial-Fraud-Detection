import numpy as np
from Modules.load_encodings import get_encodings

def cleaned_data(data):
    data.drop(columns=["Name Origin", "Name Destination", "isFlaggedFraud", "Date of Transaction"], inplace=True)
    data.set_index('Transaction ID', inplace=True)
    type_encoding, acctype_encoding, timeofday_encoding, branch_encoding = get_encodings()
    data['Type'] = data['Type'].map(type_encoding)
    data['Account Type'] = data['Account Type'].map(acctype_encoding)
    data['Time of Day'] = data['Time of Day'].map(timeofday_encoding)
    data['Branch'] = data['Branch'].map(branch_encoding)

    return data