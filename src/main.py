from model import *
import numpy as np

model = load_model("./models/OSM-0.1.orcamodel")

# print("Input embedding lookup shape:", len(model.input_embedding_lookup))
# print("Query embedding matrix shape:", np.shape(model.query_embedding_matrix))
# print("Key embedding matrix shape:", np.shape(model.key_embedding_matrix))
# print("Value embedding matrix shape:", np.shape(model.value_embedding_matrix))
# print("Output embedding matrix shape:", np.shape(model.output_embedding_matrix))
# print("Vocab size:", model.vocab_size)
# print("Embedding dim:", model.embedding_dim)
# print("Context length:", model.context_length)
# print("Model id:", model.model_id)

def gen(seed, length, temp):
    for _ in range(length):
        seed += model.generate_word(seed, temp)
    return seed

print(gen("the ", 10, 0.5))