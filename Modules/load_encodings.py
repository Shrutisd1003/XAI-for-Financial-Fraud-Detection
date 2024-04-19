import ast

def read_encoding(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readline()

def get_encodings():
    encodings = {}

    encoding_files = ['type_encoding.txt', 'acctype_encoding.txt', 'timeofday_encoding.txt', 'branch_encoding.txt']

    for file_name in encoding_files:
        key = file_name.split('_')[0] + '_encoding'
        encodings[key] = read_encoding(f'Model building/Encodings/{file_name}')

    type_encoding = ast.literal_eval(encodings['type_encoding'])
    acctype_encoding = ast.literal_eval(encodings['acctype_encoding'])
    timeofday_encoding = ast.literal_eval(encodings['timeofday_encoding'])
    branch_encoding = ast.literal_eval(encodings['branch_encoding'])

    return type_encoding, acctype_encoding, timeofday_encoding, branch_encoding