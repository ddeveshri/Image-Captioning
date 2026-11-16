import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .config import CONFIG
from .model import ImageCaptioner
from .vocabulary import Vocabulary

def main():
    parser = argparse.ArgumentParser(description="Generate a caption for one image.")
    parser.add_argument("--image", required=True, help="Path to an image file")
    args = parser.parse_args()

    checkpoint = torch.load(CONFIG.output_dir / "best_model.pth", map_location=CONFIG.device)
    vocab = Vocabulary.from_dict(checkpoint["vocab"])
    params = checkpoint["config"]
    model = ImageCaptioner(len(vocab), params["embedding_dim"],
                           params["hidden_dim"], params["dropout"]).to(CONFIG.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((params["image_size"], params["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(CONFIG.device)
    with torch.no_grad():
        ids = model.generate(image, vocab, params["max_caption_length"])[0]
    words = []
    for idx in ids.tolist():
        word = vocab.itos[idx]
        if word == "<end>":
            break
        if word not in ("<pad>", "<start>"):
            words.append(word)
    print("Generated caption:", " ".join(words))

if __name__ == "__main__":
    main()
