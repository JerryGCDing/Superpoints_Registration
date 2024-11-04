import pickle


def dump_pickle(data, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
