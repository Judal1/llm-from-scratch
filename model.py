import random
from embedding import get_embedded
from tokenizer import encode

def create_weights(embedding_dim, vocab_size):
    weight = [
        [random.uniform(-1,1) for _ in range(vocab_size)]
        for _ in range(embedding_dim)
    ]
    
    return weight

def forward_pass(embedded_sequence, weights):
    embedding_dim = len(weights)
    vocab_size = len(weights[0])
    score_vector = [0.0] * vocab_size

    for seq in embedded_sequence:
        for col_idx in range(vocab_size):
            dot = sum(seq[row_idx] * weights[row_idx][col_idx] for row_idx in range(embedding_dim))
            score_vector[col_idx] += dot

    return score_vector

def predict_next_char(score_vector):
    max_index = score_vector.index(max(score_vector))
    return max_index

def generate_text(start_sequence, weights, embedding_dict, itos,stoi, length, window_size):
    final_sequence = start_sequence
    for _ in range(length):
        window_sequence = final_sequence[-window_size:] if window_size > 0 else final_sequence
        indexes = encode(window_sequence,stoi)
        embedded = get_embedded(indexes,embedding_dict)
        fw = forward_pass(embedded, weights)
        print(fw)
        index = predict_next_char(fw)
        final_sequence += itos[index]
    return final_sequence