from tokenizer import build_vocab, encode, decode

def main():
    text = "hello world hello"
    stoi, itos = build_vocab(text)
    print("Vocabulary:", stoi)
    encoded = encode(text, stoi)
    print("Encoded:", encoded)
    decoded = decode(encoded, itos)
    print("Decoded:", decoded)

if __name__ == "__main__":
    main()