from tokenizer import build_vocab, encode, decode
from embedding import create_embedding, get_embedded

def main():
    text = "hello"
    stoi, itos = build_vocab(text)
    print("Vocabulary:", stoi)
    encoded = encode(text, stoi)
    print("Encoded:", encoded)
    decoded = decode(encoded, itos)
    print("Decoded:", decoded)
    embedding = create_embedding(stoi, 4)
    print('Embedding = ' , embedding)
    embedded = get_embedded(encoded,embedding)
    print("Embedded = " , embedded)
    embedded = get_embedded(encoded,embedding)
    print("Embedded = " , embedded)

if __name__ == "__main__":
    main()