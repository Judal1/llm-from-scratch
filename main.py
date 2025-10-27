from tokenizer import build_vocab, encode, decode
from embedding import create_embedding, get_embedded
from model import create_weights, forward_pass, predict_next_char, generate_text

EMBEDDING_SIZE = 4

def main():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam consequat tempus massa ut vehicula. Integer est nulla, condimentum ut magna quis, semper mattis libero. Vivamus vel urna non eros fermentum luctus ac vel nunc. Aliquam condimentum, justo ut fringilla sodales, nisi dui molestie eros, efficitur eleifend massa libero a purus. Fusce vitae rutrum leo, eget porta orci. Nulla ut eros ut tortor porttitor feugiat sit amet ac est. Sed elementum eleifend nulla fringilla aliquet. Vivamus ultrices arcu ac sollicitudin fringilla. Quisque convallis quam eu aliquam blandit."
    stoi, itos = build_vocab(text)
    print("Vocabulary:", stoi)
    encoded = encode(text, stoi)
    decoded = decode(encoded, itos)
    embedding = create_embedding(stoi, EMBEDDING_SIZE)
    embedded = get_embedded(encoded,embedding)
    weights = create_weights(EMBEDDING_SIZE, len(stoi))
    score_vector = forward_pass(embedded,weights)
    predicted_index = predict_next_char(score_vector)
    
    
    print("the predicted index is = " , itos[predicted_index])
    generated_text = generate_text(text, weights,embedding,itos,stoi,200, 20)
    print("from ", text, " we generated : " , generated_text)

if __name__ == "__main__":
    main()