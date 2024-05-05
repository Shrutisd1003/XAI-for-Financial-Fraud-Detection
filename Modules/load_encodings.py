import ast

def read_encoding(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readline()

def get_encodings():
    encodings = {}

    encoding_files = ['branch_encoding.txt', 'sender_encoding.txt', 'receiver_encoding.txt']

    for file_name in encoding_files:
        key = file_name.split('_')[0] + '_encoding'
        encodings[key] = read_encoding(f'Model building/Encodings/{file_name}')

    branch_encoding = ast.literal_eval(encodings['branch_encoding'])
    sender_encoding = ast.literal_eval(encodings['sender_encoding'])
    receiver_encoding = ast.literal_eval(encodings['receiver_encoding'])

    return  branch_encoding, sender_encoding, receiver_encoding