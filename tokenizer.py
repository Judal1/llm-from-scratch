import re
def tokenize(text):
    """
    Retourne la liste de tokens en séparant mots et ponctuation.
    Exemple: "Bonjour, ça va ?" -> ["Bonjour", ",", "ça", "va", "?"]
    """
    return re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)

def build_vocab(text):
    words = tokenize(text)
    tokens = sorted(set(words))
    stoi = {word: i for i, word in enumerate(tokens)}
    itos = {i: word for i, word in enumerate(tokens)}
    return stoi, itos

def encode(text, stoi):
    return [stoi[word] for word in tokenize(text) if word in stoi]

def decode(tokens, itos):
    # Reconstruire la chaîne en collant la ponctuation au mot précédent
    words = [itos[t] for t in tokens if t in itos]
    out = ""
    for w in words:
        if re.match(r"[^\w\s]", w):  # ponctuation -> coller sans espace
            out += w
        else:
            if out and not out.endswith(" "):
                out += " "
            out += w
    return out

def build_vocab_char(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos
def encode_char(text, stoi):
    return [stoi[ch] for ch in text if ch in stoi]
def decode_char(tokens, itos):
    return ''.join(itos[t] for t in tokens if t in itos)