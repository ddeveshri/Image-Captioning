from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    # Expected Flickr8k layout:
    # data/Flickr8k/Images/*.jpg
    # data/Flickr8k/captions.txt OR Flickr8k.token.txt
    data_dir: Path = Path("data/Flickr8k")
    image_dir: Path = Path("data/Flickr8k/Images")
    captions_file: Path = Path("data/Flickr8k/captions.txt")
    output_dir: Path = Path("outputs")
    image_size: int = 224
    embedding_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.3
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    min_word_freq: int = 2
    max_caption_length: int = 30
    train_split: float = 0.8
    val_split: float = 0.1
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = Config()
