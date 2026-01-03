"""
Validation Utilities
"""

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def validate(model, loader, criterion, device):
    """
    Validate model on validation set
    
    Args:
        model: PyTorch model
        loader: DataLoader
        criterion: Loss function
        device: torch.device
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy (%)
        balanced_acc: Balanced accuracy
        all_preds: List of predictions
        all_labels: List of true labels
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            mri = batch['mri'].to(device)
            cog_seq = batch['cognitive_seq'].to(device)
            cog_static = batch['cognitive_static'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(mri, cog_seq, cog_static)
            loss = criterion(outputs, labels)

            # Collect predictions
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return (
        total_loss / len(loader),
        acc,
        balanced_acc,
        all_preds,
        all_labels
    )