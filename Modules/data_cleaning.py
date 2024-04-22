from Modules.load_encodings import get_encodings

def cleaned_data(data):
    transaction_ids = data["Transaction ID"]
    data['Receiver'] = 'Customer'
    data.loc[data['Name Destination'].str.contains('M'), 'Receiver'] = 'Merchant'
    data.loc[data['Name Destination'].str.contains('C'), 'Receiver'] = 'Customer'
    data.loc[data['Name Origin'] == data['Name Destination'], 'Receiver'] = 'Self'
    data.drop(columns=['Transaction ID', "Name Origin", "isFlaggedFraud", "Date of Transaction", "Name Destination"], inplace=True)
    
    rec = data.pop("Receiver")
    data.insert(7, "Receiver", rec)
    
    data['isOriginInconsistent'] = (abs(data['Old Balance Origin'] - data['New Balance Origin']).round(2) != data['Amount']).astype(int)
    data['isDestinationInconsistent'] = 0
    data.loc[data['Type'].isin(['PAYMENT', 'TRANSFER', 'CREDIT']), 'isDestinationInconsistent'] = (abs(data['Old Balance Destination'] - data['New Balance Destination']).round(2) != data['Amount']).astype(int)
    data.loc[data['Type'].isin(['CASH_IN', 'CASH_OUT']), 'isDestinationInconsistent'] = ((data['New Balance Destination'].round(2) == data['Old Balance Destination'].round(2)) & (data['New Balance Destination'].round(2) != 0.00)).astype(int)
    
    ori = data.pop("isOriginInconsistent")
    data.insert(10, "isOriginInconsistent", ori)

    dest = data.pop("isDestinationInconsistent")
    data.insert(11, "isDestinationInconsistent", dest)

    type_encoding, acctype_encoding, timeofday_encoding, branch_encoding, receiver_encoding = get_encodings()
    data['Type'] = data['Type'].map(type_encoding)
    data['Account Type'] = data['Account Type'].map(acctype_encoding)
    data['Time of Day'] = data['Time of Day'].map(timeofday_encoding)
    data['Branch'] = data['Branch'].map(branch_encoding)
    data['Receiver'] = data['Receiver'].map(receiver_encoding)

    return transaction_ids, data