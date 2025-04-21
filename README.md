# Audio Classification using ResNet

I've been experimenting with audio classification tasks, mainly because I have never done it before. On the way needed to build a really sample efficient model. So see this project as a collection of practical tricks and techniques to get strong performance out of a model using limited training data.


## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python train.py
```

## Model Architecture

ResNet-18 backbone with the following modifications:
- First convolutional layer adapted for single-channel input
- Added dropout layers for regularization
- Modified fully connected layers with additional dropout

## Training

The training process includes:
- Label smoothing for better generalization
- Learning rate scheduling
- Early stopping
- Mixup augmentation
- Gradient clipping

## Results

The model achieves competitive results on the ESC-50 (test accuracy >80%) with limited training data.

## References

- Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. In *Proceedings of the 23rd Annual ACM Conference on Multimedia* (pp. 1015-1018). ACM Press. https://doi.org/10.1145/2733373.2806390
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *arXiv preprint arXiv:1512.03385*. https://arxiv.org/abs/1512.03385