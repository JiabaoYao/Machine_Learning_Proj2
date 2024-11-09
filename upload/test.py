import pickle

with open('params_result.pickle', 'rb') as f:
    params = pickle.load(f)
    print(params)