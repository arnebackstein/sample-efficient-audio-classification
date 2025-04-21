import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_spectrogram(spec, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.numpy(), aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'r-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    if labels.dim() > 1:
        correct = (predicted == torch.argmax(labels, dim=1)).sum().item()
    else:
        correct = (predicted == labels).sum().item()
    return 100 * correct / total 