from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, Dataset
from functools import cache
import torch


def get_transforms(config):
    if config['augmentation'] == 'same':
        return transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])

    elif config['augmentation'] == 'basic':
        return transforms.Compose([
            transforms.ToImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])
    elif config['augmentation'] == 'advanced':
        return transforms.Compose([
            transforms.ToImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(32),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])
    elif config['augmentation'] == 'test':
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])
    else:
        return transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])


class CachedDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        if transform is None:
            self.transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
        else:
            self.transform = transform

    @cache
    def _load_data(self, idx):
        data, target = self.base_dataset[idx]
        return data, target

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, target = self._load_data(idx)

        if self.transform:
            data = self.transform(data)

        return data, target


def get_dataset(config):
    transform = get_transforms(config)
    if config['dataset'] == 'MNIST':
        train_dataset = datasets.MNIST(root='data/', train=True, download=True)
        test_dataset = datasets.MNIST(root='data/', train=False, download=True)
    elif config['dataset'] == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='data/', train=True, download=True)
        test_dataset = datasets.CIFAR10(root='data/', train=False, download=True)
    elif config['dataset'] == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='data/', train=True, download=True)
        test_dataset = datasets.CIFAR100(root='data/', train=False, download=True)
    else:
        raise ValueError("Dataset not supported.")

    return CachedDataset(train_dataset, transform=transform), CachedDataset(test_dataset)


def get_dataloaders(config):
    train_dataset, val_dataset = get_dataset(config)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader
