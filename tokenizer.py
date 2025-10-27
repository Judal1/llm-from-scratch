def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos
def encode(text, stoi):
    return [stoi[ch] for ch in text if ch in stoi]
def decode(tokens, itos):
    return ''.join(itos[t] for t in tokens if t in itos)
