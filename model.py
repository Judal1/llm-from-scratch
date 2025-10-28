import random
from embedding import get_embedded
from tokenizer import encode

def create_weights(embedding_dim, vocab_size):
    weight = [
        [random.uniform(-1,1) for _ in range(embedding_dim)]
        for _ in range(vocab_size)
    ]
    
    return weight

def forward_pass(embedded_sequence, weights):
    vocab_size = len(weights)
    embedding_dim = len(weights[0])
    score_vector = [0.0] * vocab_size

    for seq in embedded_sequence:  # seq est un vecteur d'embedding
        for token_idx in range(vocab_size):
            dot = sum(seq[d] * weights[token_idx][d] for d in range(embedding_dim))
            score_vector[token_idx] += dot

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
        index = predict_next_char(fw)
        final_sequence += itos[index]
    return final_sequence

def update_weights(weights, batch_gradient, embedded_inputs, learning_rate):
    batch_size = len(batch_gradient)
    vocab_size = len(weights)
    embedding_dim = len(weights[0])
    
    medium_grads = [
        [0 for _ in row]
        for row in weights
    ]
    for i in range(batch_size):
        grad_i = batch_gradient[i]
        embed_i = embedded_inputs[i]
        for j in range(vocab_size):
            for k in range(embedding_dim):
                medium_grads[j][k] += grad_i[j]*embed_i[k]
    for j in range(vocab_size):
        for k in range(embedding_dim):
            avg_grad = medium_grads[j][k] / batch_size
            weights[j][k] -= learning_rate * avg_grad

    return weights