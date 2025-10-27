import random

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