import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        if target.dim() == 1:
            target = torch.nn.functional.one_hot(target, num_classes=n_classes).float()
        with torch.no_grad():
            target = target * (1 - self.smoothing) + self.smoothing / n_classes
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(pred, dim=1),
            target,
            reduction='batchmean'
        )

class AudioResNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        
        for name, module in self.model.named_children():
            if isinstance(module, nn.Sequential):
                for block in module:
                    if hasattr(block, 'conv1'):
                        block.dropout = nn.Dropout(0.2)

        self.model.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x) 