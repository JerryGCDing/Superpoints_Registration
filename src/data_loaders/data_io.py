import pickle


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data
