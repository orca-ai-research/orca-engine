import logging
import re
import numpy as np

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
    
    if model is None:
        raise ValueError("Model is None")
    if input_text is None:
        raise ValueError("Input text is None")
    if temperature <= 0:
        raise ValueError("Temperature is 0 or negative")
    if attention_repition < 0:
        raise ValueError("Attention repetition is negative")

    print("Input text:", input_text)
    print("Temperature:", temperature)
    print("Attention Repetition:", attention_repition)

    try:
        embedded_input = input_embedding(input_text, model.input_embedding_lookup)
    except KeyError as e:
        logging.error(f"KeyError: {e}")
        raise

    new_embedded_input = []
    
    for i in embedded_input:
        new_i = [len(i) - 1]
        for j in i:
            new_i.append(j[0])
        new_embedded_input.append(new_i)
        
    embedded_input = new_embedded_input
    
    print("Length of embedded_input:", len(embedded_input))

    for _ in range(attention_repition):
        try:
            current_output = attention(
                embedded_input,
                model.query_embedding_matrix,
                model.key_embedding_matrix,
                model.value_embedding_matrix,
                temperature,
            )
        except Exception as e:
            logging.error(f"Exception in attention: {e}")
            raise

    print("Length of current_output:", len(current_output))

    output = output_embedding(current_output, model.output_embedding_matrix)
    output_weights = softmax(output[len(output) - 1], temperature)

    print("Length of output:", len(output))
    print("Length of output_weights:", len(output_weights))

    return output_weights


def input_embedding(input_text: str, input_embeddings):
    tokens = re.findall(r"\b\w+\b|[^\w\s{}|]|[{|ENDOFTEXT|}]", input_text.lower())

    if input_embeddings is None:
        raise ValueError("Input embeddings is None")

    embeddings = [input_embeddings.get(token, [0] * 1228) for token in tokens]
    return embeddings

def softmax(arr, temperature):
    if temperature == 0:
        raise ValueError("Temperature cannot be 0")

    try:
        return np.exp(np.array(arr) / temperature) / np.sum(
            np.exp(np.array(arr) / temperature)
        )
    except Exception as e:
        logging.error(f"Exception in softmax: {e}")
        raise


def get_queries(input_seq, query_embeddings):
    if query_embeddings is None:
        raise ValueError("Query embeddings is None")

    return [np.matmul(query_embeddings, input_vec) for input_vec in input_seq]


def get_keys(input_seq, key_embeddings):
    if key_embeddings is None:
        raise ValueError("Key embeddings is None")

    return [np.matmul(key_embeddings, input_vec) for input_vec in input_seq]


def calculate_attention(queries, keys):
    return [[query * key for key in keys] for query in queries]


def normalize_attention(attention, temperature):
    if attention is None:
        raise ValueError("Attention is None")

    normalized_attention = []
    for row in attention:
        normalized_row = []
        for element in row:
            normalized_row.append(element / temperature)
        normalized_attention.append(normalized_row)
    return normalized_attention


def apply_attention(input_seq, attention, value_embeddings):
    if value_embeddings is None:
        raise ValueError("Value embeddings is None")
    if attention is None:
        raise ValueError("Attention is None")

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
    if query_embeddings is None:
        raise ValueError("Query embeddings is None")
    if key_embeddings is None:
        raise ValueError("Key embeddings is None")
    if value_embeddings is None:
        raise ValueError("Value embeddings is None")

    queries = get_queries(input_seq, query_embeddings)
    keys = get_keys(input_seq, key_embeddings)
    attention = calculate_attention(queries, keys)
    normalized_attention = normalize_attention(attention, temperature)
    output = apply_attention(input_seq, normalized_attention, value_embeddings)

    return output

def output_embedding(current_output, output_embeddings):
    if output_embeddings is None:
        raise ValueError("Output embeddings is None")

    return np.matmul(output_embeddings, current_output)
