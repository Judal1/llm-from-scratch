import random

def create_embedding(stoi,vector_size):
    vocab_size = max(stoi.values()) + 1 if stoi else 0
    
    embedding = [
        [random.uniform(-1, 1) for _ in range(vector_size)]
        for _ in range(vocab_size)
    ]
    return embedding
def get_embedded(indexes, embedding):
    embedded = [embedding[index] for index in indexes]
    return embedded