from lib.engine import generate_word
import pickle
import random
from lib.vocab_reader import get_vocab

embedding_dim = 1228
vocab_size = 3668
context_length = 1000

class Model:
    def __init__(
        self,
        input_embedding_lookup: dict,
        query_embedding_matrix: list,
        key_embedding_matrix: list,
        value_embedding_matrix: list,
        output_embedding_matrix: list,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        model_id: str,
    ):
        self.input_embedding_lookup = input_embedding_lookup
        self.query_embedding_matrix = query_embedding_matrix
        self.key_embedding_matrix = key_embedding_matrix
        self.value_embedding_matrix = value_embedding_matrix
        self.output_embedding_matrix = output_embedding_matrix
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.model_id = model_id

    def generate_word(self, input_text, temperature, attention_repition=1):
        return generate_word(input_text, self, temperature, attention_repition)


def rand_arr(dim):
    if len(dim) == 1:
        return [random.random() for _ in range(dim[0])]
    elif len(dim) == 2:
        return [[random.random() for _ in range(dim[1])] for _ in range(dim[0])]
    else:
        raise ValueError("Invalid dimension: must be 1 or 2 dimensions")


def make_model(model_id):
    vocab = get_vocab()

    embedding_dict = {}

    for i in range(vocab_size):
        embedding_dict[vocab[i]] = rand_arr((embedding_dim, 1))

    query_embedding_matrix = rand_arr((embedding_dim, embedding_dim))
    key_embedding_matrix = rand_arr((embedding_dim, embedding_dim))
    value_embedding_matrix = rand_arr((embedding_dim, embedding_dim))
    output_embedding_matrix = rand_arr((embedding_dim, embedding_dim))

    return Model(
        embedding_dict,
        query_embedding_matrix,
        key_embedding_matrix,
        value_embedding_matrix,
        output_embedding_matrix,
        vocab_size,
        embedding_dim,
        context_length,
        model_id,
    )


def save_model(model: Model, filename: str):
    try:
        with open(filename, "wb") as file:
            pickle.dump(model, file)
        print(f"Model saved successfully to {filename}")
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")


def load_model(filename: str) -> Model:
    try:
        with open(filename, "rb") as file:
            obj = pickle.load(file)
        print(f"Model loaded successfully from {filename}")
        return obj
    except Exception as e:
        print(f"Error occurred while loading the model: {e}")
        return None
