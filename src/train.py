import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .config import CONFIG
from .data import build_loaders
from .model import ImageCaptioner

def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, captions, _ in tqdm(loader, leave=False):
            images = images.to(device)
            captions = captions.to(device)
            logits = model(images, captions)
            targets = captions[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            token_count = targets.ne(0).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count
    return total_loss / max(total_tokens, 1)

def main():
    cfg = CONFIG
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, _, vocab = build_loaders(cfg)
    model = ImageCaptioner(len(vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout).to(cfg.device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate
    )
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, cfg.device, True)
        val_loss = run_epoch(model, val_loader, criterion, None, cfg.device, False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab.to_dict(),
                "config": {
                    "embedding_dim": cfg.embedding_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "dropout": cfg.dropout,
                    "image_size": cfg.image_size,
                    "max_caption_length": cfg.max_caption_length,
                }
            }, cfg.output_dir / "best_model.pth")

    with (cfg.output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"Best checkpoint saved to {cfg.output_dir / 'best_model.pth'}")

if __name__ == "__main__":
    main()
