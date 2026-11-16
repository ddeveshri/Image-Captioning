import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .vocabulary import Vocabulary

def read_captions(captions_file):
    """Read either Flickr8k.token.txt or a CSV with image,caption columns."""
    path = Path(captions_file)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        handle.seek(0)
        if "," in first and ("image" in first.lower() or "caption" in first.lower()):
            reader = csv.DictReader(handle)
            for row in reader:
                image_name = row.get("image") or row.get("filename") or row.get("Image")
                caption = row.get("caption") or row.get("Caption")
                if image_name and caption:
                    records.append((Path(image_name).name, caption.strip()))
        else:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                # Flickr8k token format: image.jpg#0<TAB>caption
                parts = line.split("\\t", 1)
                if len(parts) == 2:
                    records.append((parts[0].split("#")[0], parts[1].strip()))
    if not records:
        raise ValueError(f"No captions found in {path}. Check the file format.")
    return records

def split_by_image(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    grouped = defaultdict(list)
    for image, caption in records:
        grouped[image].append(caption)
    image_names = sorted(grouped)
    random.Random(seed).shuffle(image_names)
    n = len(image_names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_names = set(image_names[:n_train])
    val_names = set(image_names[n_train:n_train+n_val])
    train = [(name, cap) for name in train_names for cap in grouped[name]]
    val = [(name, cap) for name in val_names for cap in grouped[name]]
    test = [(name, cap) for name in image_names[n_train+n_val:] for cap in grouped[name]]
    return train, val, test

class CaptionDataset(Dataset):
    def __init__(self, records, image_dir, vocab, image_size=224, max_length=30, train=False):
        self.records = records
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_name, caption = self.records[index]
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        tokens = [self.vocab.stoi["<start>"]]
        tokens += self.vocab.numericalize(caption)
        tokens = tokens[:self.max_length - 1]
        tokens += [self.vocab.stoi["<end>"]]
        return image, torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    lengths = [len(c) for c in captions]
    padded = torch.nn.utils.rnn.pad_sequence(
        captions, batch_first=True, padding_value=0
    )
    return images, padded, torch.tensor(lengths)

def build_loaders(config):
    records = read_captions(config.captions_file)
    train_records, val_records, test_records = split_by_image(
        records, config.train_split, config.val_split, config.seed
    )
    vocab = Vocabulary(config.min_word_freq)
    vocab.build([caption for _, caption in train_records])

    common = dict(image_dir=config.image_dir, vocab=vocab,
                  image_size=config.image_size,
                  max_length=config.max_caption_length)
    train_ds = CaptionDataset(train_records, **common)
    val_ds = CaptionDataset(val_records, **common)
    test_ds = CaptionDataset(test_records, **common)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, vocab
