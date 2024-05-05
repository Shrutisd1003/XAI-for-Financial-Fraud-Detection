from Modules.load_encodings import get_encodings

def cleaned_data(data):
    transaction_ids = data["Transaction ID"]
    data.drop(columns=['Transaction ID', 'Time of Transaction'], inplace=True)
    
    branch_encoding, sender_encoding, receiver_encoding = get_encodings()
    data['Branch'] = data['Branch'].map(branch_encoding)
    data['Sender'] = data['Sender'].map(sender_encoding)
    data['Receiver'] = data['Receiver'].map(receiver_encoding)

    return transaction_ids, data