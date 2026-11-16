import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from .config import CONFIG
from .data import build_loaders
from .model import ImageCaptioner
from .vocabulary import Vocabulary

def decode(ids, vocab):
    words = []
    for idx in ids:
        word = vocab.itos[int(idx)]
        if word == "<end>":
            break
        if word not in ("<pad>", "<start>"):
            words.append(word)
    return " ".join(words)

def main():
    cfg = CONFIG
    _, _, test_loader, _ = build_loaders(cfg)
    checkpoint = torch.load(cfg.output_dir / "best_model.pth", map_location=cfg.device)
    vocab = Vocabulary.from_dict(checkpoint["vocab"])
    params = checkpoint["config"]
    model = ImageCaptioner(len(vocab), params["embedding_dim"],
                           params["hidden_dim"], params["dropout"]).to(cfg.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    references, hypotheses = [], []
    examples = []
    with torch.no_grad():
        for images, captions, _ in tqdm(test_loader):
            images = images.to(cfg.device)
            predictions = model.generate(images, vocab, params["max_caption_length"])
            for reference, prediction in zip(captions, predictions):
                refs = [decode(reference, vocab).split()]
                hyp = decode(prediction, vocab).split()
                references.append(refs)
                hypotheses.append(hyp)
                if len(examples) < 8:
                    examples.append((refs[0], " ".join(hyp)))

    bleu = corpus_bleu(references, hypotheses,
                       smoothing_function=SmoothingFunction().method1)
    print(f"Test corpus BLEU-4: {bleu:.4f}")
    print("\\nSample predictions:")
    for actual, predicted in examples:
        print(f"Reference:  {actual}\\nPredicted:  {predicted}\\n")

    history_path = cfg.output_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Token-level cross-entropy loss")
        plt.title("Training history")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.output_dir / "loss_curve.png", dpi=150)
        print(f"Loss curve saved to {cfg.output_dir / 'loss_curve.png'}")

if __name__ == "__main__":
    main()
