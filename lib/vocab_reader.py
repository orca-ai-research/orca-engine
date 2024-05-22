import os

def get_vocab():
    with open(os.path.join(os.path.dirname(__file__), "vocab.txt"), "r") as file:
        vocab = file.read().splitlines()
    return vocab