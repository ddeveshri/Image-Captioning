# Image Captioning using CNN + LSTM

A deep learning project that generates meaningful natural-language descriptions of images using a CNN–LSTM encoder-decoder architecture. The model extracts visual features using ResNet50 and generates captions using an LSTM decoder.

## Project Overview

Image captioning combines Computer Vision and Natural Language Processing to understand image content and generate relevant textual descriptions.

Key Features
- Extracts image features using a pretrained ResNet50 CNN.
- Generates captions using an LSTM-based decoder.
- Preprocesses captions through tokenization and vocabulary mapping.
- Implements teacher forcing for sequence training.
- Evaluates generated captions using BLEU-4 scores.

## Tech Stack

Programming Language - Python 
Deep Learning - PyTorch
CNN Architecture - ResNet50
Sequence Modeling - LSTM 
Image Processing - PIL, torchvision 
NLP Preprocessing - Tokenization, Vocabulary Mapping 
Dataset - Flickr8k 
Evaluation - BLEU-4 
Visualization - Matplotlib 
Development Environment - Google Colab, Jupyter Notebook 

## Dataset

Flickr8k Dataset**

- 8,000 images with 5 human-written captions per image.
- Used for training, validation, and testing.
- Captions processed through lowercasing, tokenization, and vocabulary construction.
- Images resized and normalized before feature extraction.

## Model Architecture

The project follows a CNN–LSTM encoder-decoder architecture.

1. CNN Encoder (ResNet50): Extracts visual features from input images using pretrained ImageNet weights.
2. Feature Projection: Maps extracted image features into a 256-dimensional representation.
3. LSTM Decoder: Uses image features to initialize its hidden and cell states and generates captions sequentially.
4. Caption Generation: Predicts the next word until the end-of-sequence token is generated.

## Implementation

- Preprocessed Flickr8k images and captions.
- Built a custom vocabulary with word-to-index mapping.
- Implemented a ResNet50-based CNN encoder and LSTM decoder.
- Trained the model using teacher forcing, cross-entropy loss, and the Adam optimizer.
- Monitored training and validation loss.
- Generated captions for unseen images.
- Evaluated caption generation using BLEU-4.

## Evaluation

Model performance is evaluated using:

- BLEU-4 Score: Measures the overlap of generated captions with reference captions.
- Training and Validation Loss: Tracks model learning and generalization.
- Qualitative Evaluation: Compares generated captions with actual image descriptions.

## Results

Training and evaluation outputs include:

- Training and validation loss curves.
- Generated image-caption predictions.
- Actual versus predicted caption comparisons.
- BLEU-4 evaluation score.

## Future Enhancements

- Implement attention mechanisms for improved visual feature selection.
- Explore beam search for caption generation.
- Fine-tune pretrained CNN layers.
- Compare different CNN architectures.
- Explore additional evaluation metrics such as METEOR and CIDEr.

## Project Domain

Deep Learning | Computer Vision | Natural Language Processing | Sequence Modeling
