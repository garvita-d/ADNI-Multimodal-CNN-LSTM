"""
Training Script with Cross-Validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        loader: DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
    
    Returns:
        avg_loss: Average loss
        accuracy: Training accuracy (%)
    """
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc='  Training', leave=False)
    for batch in pbar:
        mri = batch['mri'].to(device)
        cog_seq = batch['cognitive_seq'].to(device)
        cog_static = batch['cognitive_static'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(mri, cog_seq, cog_static)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

    return total_loss / len(loader), 100. * correct / total


def get_balanced_sampler(records):
    """
    Create WeightedRandomSampler for class balancing
    
    Args:
        records: List of dataset records
    
    Returns:
        WeightedRandomSampler
    """
    labels = [r['label'] for r in records]
    class_counts = Counter(labels)
    
    # Compute sample weights
    weights = [1.0 / class_counts[l] for l in labels]
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


def get_class_weights(records, num_classes=3, device='cpu'):
    """
    Compute class weights for weighted loss
    
    Args:
        records: List of dataset records
        num_classes: Number of classes
        device: torch.device
    
    Returns:
        Tensor of class weights
    """
    labels = [r['label'] for r in records]
    class_counts = Counter(labels)
    
    # Inverse frequency weighting
    weights = torch.tensor([
        1.0 / class_counts.get(i, 1) for i in range(num_classes)
    ], device=device)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return weights