"""
Evaluation Metrics and Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names ['CN', 'MCI', 'AD']
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_classification_metrics(y_true, y_pred, class_names):
    """
    Print classification report and balanced accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    
    # Overall accuracy
    acc = 100 * np.mean(np.array(y_pred) == np.array(y_true))
    print(f"Overall Accuracy: {acc:.2f}%")
    
    # Balanced accuracy
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3
    ))


def plot_cv_results(fold_scores, save_path=None):
    """
    Plot cross-validation results
    
    Args:
        fold_scores: List of balanced accuracy scores per fold
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    folds = list(range(1, len(fold_scores) + 1))
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    plt.bar(folds, fold_scores, alpha=0.7, color='steelblue')
    plt.axhline(
        y=mean_score,
        color='red',
        linestyle='--',
        label=f'Mean: {mean_score:.3f} ± {std_score:.3f}'
    )
    
    plt.xlabel('Fold')
    plt.ylabel('Balanced Accuracy')
    plt.title('Cross-Validation Results')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()