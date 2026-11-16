import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class CNNEncoder(nn.Module):
    """Pretrained ResNet50 image feature extractor with a learnable projection."""
    def __init__(self, embed_dim, fine_tune=False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        for parameter in self.features.parameters():
            parameter.requires_grad = fine_tune
        self.projection = nn.Linear(backbone.fc.in_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, images):
        features = self.features(images).flatten(1)
        return self.bn(self.projection(features))

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_features, captions):
        embeddings = self.dropout(self.embedding(captions))
        h0 = torch.tanh(self.init_h(image_features)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(image_features)).unsqueeze(0)
        outputs, _ = self.lstm(embeddings, (h0, c0))
        return self.output(self.dropout(outputs))

class ImageCaptioner(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim)
        self.decoder = CaptionDecoder(vocab_size, embed_dim, hidden_dim, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        # Input excludes final token; targets are shifted by one in training.
        return self.decoder(features, captions[:, :-1])

    @torch.no_grad()
    def generate(self, images, vocab, max_length=30):
        self.eval()
        features = self.encoder(images)
        batch_size = images.size(0)
        device = images.device
        h = torch.tanh(self.decoder.init_h(features)).unsqueeze(0)
        c = torch.tanh(self.decoder.init_c(features)).unsqueeze(0)
        current = torch.full((batch_size, 1), vocab.stoi["<start>"],
                             dtype=torch.long, device=device)
        generated = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_length):
            embedded = self.decoder.embedding(current)
            output, (h, c) = self.decoder.lstm(embedded, (h, c))
            logits = self.decoder.output(output.squeeze(1))
            next_token = logits.argmax(dim=1)
            next_token = torch.where(finished,
                                     torch.full_like(next_token, vocab.stoi["<pad>"]),
                                     next_token)
            generated.append(next_token)
            finished |= next_token.eq(vocab.stoi["<end>"])
            current = next_token.unsqueeze(1)
            if finished.all():
                break
        return torch.stack(generated, dim=1)
