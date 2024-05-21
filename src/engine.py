import numpy as np
import re

def verify_model(model):
    if model is None:
        raise ValueError("Model is None")

    if model.vocab_size <= 0:
        raise ValueError("Model vocab size is 0 or negative")

    if model.embedding_dim <= 0:
        raise ValueError("Model embedding dim is 0 or negative")

    if model.context_length <= 0:
        raise ValueError("Model context length is 0 or negative")
    
    if model.model_id is None:
        raise ValueError("Model model_id is None")
    
    if model.input_embedding_lookup is None:
        raise ValueError("Model input_embedding_lookup is None")
    
    if model.query_embedding_matrix is None:
        raise ValueError("Model query_embedding_matrix is None")
    
    if model.key_embedding_matrix is None:
        raise ValueError("Model key_embedding_matrix is None")
    
    if model.value_embedding_matrix is None:
        raise ValueError("Model value_embedding_matrix is None")
    
    if model.output_embedding_matrix is None:
        raise ValueError("Model output_embedding_matrix is None")
    
    if len(model.input_embedding_lookup) is not model.vocab_size:
        raise ValueError("Model input_embedding_lookup length is not equal to vocab_size")
    
    if np.shape(model.query_embedding_matrix) is not (model.embedding_dim, model.embedding_dim):
        raise ValueError("Model query_embedding_matrix length is not equal to embedding_dim")
    
    if np.shape(model.key_embedding_matrix) is not (model.embedding_dim, model.embedding_dim):
        raise ValueError("Model key_embedding_matrix length is not equal to embedding_dim")
    
    if np.shape(model.value_embedding_matrix) is not (model.embedding_dim, model.embedding_dim):
        raise ValueError("Model value_embedding_matrix length is not equal to embedding_dim")
    
    if np.shape(model.output_embedding_matrix) is not (model.embedding_dim, model.embedding_dim):
        raise ValueError("Model output_embedding_matrix length is not equal to embedding_dim")

def output_embedding(input_seq, output_embedding_matrix):
    return [np.matmul(output_embedding_matrix, input_vec) for input_vec in input_seq]

def random_choice(options, weights, embedding_dim):
    print("Length of options: ", len(options))

    if len(options) == 1:
        return options[0]

    if len(options) != len(weights):
        raise ValueError("Options and weights must have the same length")

    return np.random.choice(options, p=np.reshape(weights, (embedding_dim)))


def generate_word(
    input_text: str, model, temperature: float, attention_repition: int
) -> str:
    """
    Generates a word based on the input text.

    Args:
        input_text (str): The input text to generate the word from.
        model (Model): The model that is used to generate the word.
        temperature (float): The temperature of the softmax function, a value between 0 and 1.

    Returns:
        str: The generated word.
    """
    
    verify_model(model)
    
    print("Input text: ", input_text)
    print("Temperature: ", temperature)
    print("Attention Repetition: ", attention_repition)

    embedded_input = input_embedding(input_text, model.input_embedding_lookup)
    
    new_embedded_input = []
    
    for i in embedded_input:
        new_i = [len(i) - 1]
        for j in i:
            new_i.append(j[0])
        new_embedded_input.append(new_i)
        
    embedded_input = new_embedded_input
    
    print("Length of embedded_input: ", len(embedded_input))

    for _ in range(attention_repition):
        current_output = attention(
            embedded_input,
            model.query_embedding_matrix,
            model.key_embedding_matrix,
            model.value_embedding_matrix,
            temperature,
        )

    print("Length of current_output: ", len(current_output))

    output = output_embedding(current_output, model.output_embedding_matrix)
    output_weights = softmax(output[len(output) - 1], temperature)

    print("Length of output: ", len(output))
    print("Length of output_weights: ", len(output_weights))

    closest_words = []

    for matrix in output:
        closest_word = None
        closest_distance = np.Infinity
        for word, vector in model.input_embedding_lookup.items():
            distance = np.linalg.norm(np.array(matrix) - np.array(vector))
            if distance < closest_distance:
                closest_word = word
                closest_distance = distance
        closest_words.append(closest_word)

    return random_choice(closest_words, output_weights, model.embedding_dim)


def input_embedding(input_text: str, input_embeddings):
    tokens = re.findall(r"\b\w+\b|[^\w\s{}|]|[{|ENDOFTEXT|}]", input_text.lower())
    print("Tokens: ", tokens)
    print("Length of tokens: ", len(tokens))
    embeddings = [input_embeddings.get(token, [0] * 1228) for token in tokens]
    print("Length of embeddings: ", len(embeddings))
    print("Embeddings: ", embeddings)
    return embeddings

def softmax(arr, temperature):
    if temperature == 0:
        raise ValueError("Temperature cannot be 0")
    return np.exp(np.array(arr) / temperature) / np.sum(
        np.exp(np.array(arr) / temperature)
    )


def get_queries(input_seq, query_embeddings):
    return [np.matmul(query_embeddings, input_vec) for input_vec in input_seq]


def get_keys(input_seq, key_embeddings):
    return [np.matmul(key_embeddings, input_vec) for input_vec in input_seq]


def calculate_attention(queries, keys):
    return [[query * key for key in keys] for query in queries]


def normalize_attention(attention, temperature):
    normalized_attention = []
    for row in attention:
        normalized_row = []
        for element in row:
            normalized_row.append(element / temperature)
        normalized_attention.append(normalized_row)
    return normalized_attention


def apply_attention(input_seq, attention, value_embeddings):
    updated_attention = []
    for attention_row, input_vec in zip(attention, input_seq):
        updated_attention_row = []
        for attention_element, value_embedding in zip(attention_row, value_embeddings):
            updated_attention_row.append(
                attention_element * np.matmul(value_embedding, input_vec)
            )
        updated_attention.append(updated_attention_row)

    return updated_attention


def attention(
    input_seq,
    query_embeddings,
    key_embeddings,
    value_embeddings,
    temperature,
):
    queries = get_queries(input_seq, query_embeddings)
    keys = get_keys(input_seq, key_embeddings)
    attention = calculate_attention(queries, keys)
    normalized_attention = normalize_attention(attention, temperature)
    output = apply_attention(input_seq, normalized_attention, value_embeddings)

    return output
