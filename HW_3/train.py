import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from datasets import get_dataloaders
from models import get_model
from utils import setup_optimizer, setup_scheduler


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return val_loss / len(dataloader), 100 * correct / total


def train(config):
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_dataloaders(config)

    num_classes = 10 if config['dataset'] in ['CIFAR10', 'MNIST'] else 100
    model = get_model(config['model'], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = setup_optimizer(config, model)
    scheduler = setup_scheduler(config, optimizer)

    wandb.init(project=config['wandb_project'], config=config)
    best_val_loss = float('inf')
    patience = config['early_stopping_patience']

    print(f"Device: {device}")
    print(f"Model: {model}")
    print(f"Criterion: {criterion}")
    print(f"Optimizer: {optimizer}")
    print(f"Scheduler: {scheduler}")
    print(f"Train Loader: {train_loader}")
    print(f"Validation Loader: {val_loader}")
    print(f"Patience: {patience}")
    print(f"Best Validation Loss: {best_val_loss}")
    print(f"Config: {config}")
    print(f"Wandb Project: {config['wandb_project']}")
    print(f"Early Stopping Patience: {config['early_stopping_patience']}")
    print(f"Num Classes: {num_classes}")

    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss:
            patience -= 1
        if patience == 0:
            print("Early stopping.")
            break

        best_val_loss = min(best_val_loss, val_loss)
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
