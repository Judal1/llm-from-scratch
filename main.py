from tokenizer import build_vocab, encode, decode
from embedding import create_embedding, get_embedded
from model import create_weights, forward_pass, update_weights, predict_next_char
from math_utils import cross_entry_grad_batch
import random

EMBEDDING_SIZE = 16
LEARNING_RATE = 0.1
EPOCHS = 2000
WINDOW_SIZE = 10

def main():
    # -------------------------
    # Corpus répétitif un peu plus long
    # -------------------------
    text = (
        # motif 1
        "abcde abcde abcde abcde abcde "
        "fghij fghij fghij fghij fghij "
        "klmno klmno klmno klmno klmno "
        "pqrst pqrst pqrst pqrst pqrst "
        "uvwxy uvwxy uvwxy uvwxy uvwxy "
        # motif 2, répétition inversée
        "uvwxy uvwxy uvwxy uvwxy uvwxy "
        "pqrst pqrst pqrst pqrst pqrst "
        "klmno klmno klmno klmno klmno "
        "fghij fghij fghij fghij fghij "
        "abcde abcde abcde abcde abcde "
        # motif 3, mélangé
        "abcde fghij klmno pqrst uvwxy "
        "uvwxy pqrst klmno fghij abcde "
        "abcde fghij klmno pqrst uvwxy "
        "uvwxy pqrst klmno fghij abcde "
        "abcde fghij klmno pqrst uvwxy "
    )

    # -------------------------
    # Tokenizer
    # -------------------------
    stoi, itos = build_vocab(text)
    print("Vocabulary:", stoi)
    encoded = encode(text, stoi)
    
    # -------------------------
    # Embedding et poids
    # -------------------------
    embedding = create_embedding(stoi, EMBEDDING_SIZE)
    weights = create_weights(EMBEDDING_SIZE, len(stoi))
    
    # -------------------------
    # Sliding window pour créer séquences et targets
    # -------------------------
    sequences = []
    targets = []
    for i in range(len(encoded) - WINDOW_SIZE):
        sequences.append(encoded[i:i+WINDOW_SIZE])
        targets.append(encoded[i+WINDOW_SIZE])
    
    embedded_sequences = [get_embedded(seq, embedding) for seq in sequences]
    
    # -------------------------
    # Entraînement
    # -------------------------
    for epoch in range(EPOCHS):
        batch_grad = []
        for seq_emb, target in zip(embedded_sequences, targets):
            logits = forward_pass(seq_emb, weights)
            grad = cross_entry_grad_batch([logits], [target])[0]
            batch_grad.append(grad)
        # Update des poids sur tout le batch
        weights = update_weights(weights, batch_grad, [seq[0] for seq in embedded_sequences], LEARNING_RATE)

        # Génération rapide pour observer l’évolution
        if (epoch + 1) % 50 == 0:
            start_seq = text[:WINDOW_SIZE]
            generated = start_seq
            for _ in range(200):
                window_seq = generated[-WINDOW_SIZE:]
                indexes = encode(window_seq, stoi)
                embedded_win = get_embedded(indexes, embedding)
                score_vector = forward_pass(embedded_win, weights)
                idx = predict_next_char(score_vector)
                generated += itos[idx]
            print(f"Epoch {epoch+1} -> {generated}")

if __name__ == "__main__":
    main()
