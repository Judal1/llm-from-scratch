from tokenizer import build_vocab, encode, decode
from embedding import create_embedding, get_embedded
from model import create_weights, forward_pass, predict_next_char

EMBEDDING_SIZE = 4

def main():
    text = "hello world"
    stoi, itos = build_vocab(text)
    print("Vocabulary:", stoi)
    encoded = encode(text, stoi)
    print("Encoded:", encoded)
    decoded = decode(encoded, itos)
    print("Decoded:", decoded)
    embedding = create_embedding(stoi, EMBEDDING_SIZE)
    print('Embedding = ' , embedding)
    embedded = get_embedded(encoded,embedding)
    print("Embedded = " , embedded)
    embedded = get_embedded(encoded,embedding)
    print("Embedded = " , embedded)
    weights = create_weights(EMBEDDING_SIZE, len(stoi))
    score_vector = forward_pass(embedded,weights)
    print("Score Vector = " , score_vector)
    predicted_index = predict_next_char(score_vector)
    print("the predicted index is = " , itos[predicted_index])
    

if __name__ == "__main__":
    main()