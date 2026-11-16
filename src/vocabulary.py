import re
from collections import Counter

SPECIALS = ["<pad>", "<start>", "<end>", "<unk>"]

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = list(SPECIALS)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    @staticmethod
    def tokenize(text):
        text = text.lower()
        return re.findall(r"[a-z0-9']+", text)

    def build(self, captions):
        counts = Counter()
        for caption in captions:
            counts.update(self.tokenize(caption))
        for word, count in counts.items():
            if count >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def numericalize(self, text):
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in self.tokenize(text)]

    def __len__(self):
        return len(self.itos)

    def to_dict(self):
        return {"min_freq": self.min_freq, "itos": self.itos}

    @classmethod
    def from_dict(cls, data):
        vocab = cls(data["min_freq"])
        vocab.itos = data["itos"]
        vocab.stoi = {token: idx for idx, token in enumerate(vocab.itos)}
        return vocab
