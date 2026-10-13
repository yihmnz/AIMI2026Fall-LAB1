"""Shared dataset, ResNet, evaluation and plotting utilities."""
import random
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['NORMAL', 'PNEUMONIA']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def discover(root):
    root = Path(root)
    samples = []
    for label, name in enumerate(CLASSES):
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing class folder: {folder}')
        files = sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f'No images in {folder}')
        samples.extend((str(p), label) for p in files)
    return samples


class ChestXrayDataset(Dataset):
    """Custom Dataset: explicitly discovers files and loads RGB images."""
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert('RGB'))
        return image, label


def make_transform(size=224, training=False):
    ops = [transforms.Resize((size, size))]
    if training:
        ops += [transforms.RandomAffine(degrees=7, translate=(0.03, 0.03),
                                         scale=(0.95, 1.05), fill=0)]
    ops += [transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


def build_model(name, pretrained=True):
    builders = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
    }
    constructor, weights = builders[name]
    model = constructor(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # New randomly initialized head.
    return model


def metrics(labels, predictions):
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)), 1)
    tn, fp, fn, tp = cm.ravel()
    safe = lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0
    return {'accuracy': safe(tp + tn, cm.sum()),
            'f1': safe(2 * tp, 2 * tp + fp + fn),
            'precision': safe(tp, tp + fp), 'recall': safe(tp, tp + fn),
            'specificity': safe(tn, tn + fp), 'confusion_matrix': cm.tolist()}


def run_epoch(model, loader, device, criterion=None, optimizer=None, description=None, progress=True):
    training = optimizer is not None
    model.train(training)
    labels_all, predictions_all = [], []
    loss_sum, count, correct = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        bar = tqdm(loader, desc=description or ('Train' if training else 'Evaluate'),
                   disable=not progress, dynamic_ncols=True, leave=False)
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                if training:
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * labels.size(0)
            count += labels.size(0)
            labels_all.extend(labels.detach().cpu().tolist())
            predictions = outputs.argmax(1)
            predictions_all.extend(predictions.detach().cpu().tolist())
            correct += (predictions == labels).sum().item()
            postfix = {'acc': f'{correct / count:.4f}'}
            if criterion is not None:
                postfix['loss'] = f'{loss_sum / count:.4f}'
            bar.set_postfix(postfix)
    result = metrics(labels_all, predictions_all)
    if criterion is not None:
        result['loss'] = loss_sum / count
    return result


def plot_cm(cm, output, title):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES,
           yticklabels=CLASSES, xlabel='Predicted label', ylabel='True label', title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_histories(histories, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for name, rows in histories.items():
        for i, split in enumerate(['train', 'val']):
            for j, metric in enumerate(['accuracy', 'f1']):
                axes[i, j].plot([r['epoch'] for r in rows],
                                [r[f'{split}_{metric}'] for r in rows], label=name)
                axes[i, j].set(title=f'{split.capitalize()} {metric}',
                               xlabel='Epoch', ylabel=metric, ylim=(0, 1.01))
                axes[i, j].grid(alpha=.3)
                axes[i, j].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
