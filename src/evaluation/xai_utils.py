"""
Explainable AI (XAI) Utilities
Grad-CAM, Attention Visualization, Feature Importance
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from captum.attr import IntegratedGradients


def compute_gradcam_3d(model, mri_input, target_class):
    """
    Compute Grad-CAM for 3D MRI
    
    Args:
        model: Trained fusion model
        mri_input: MRI tensor (batch_size, 1, D, H, W)
        target_class: Target class index
    
    Returns:
        CAM heatmap (batch_size, 1, D, H, W)
    """
    model.eval()
    mri_input.requires_grad = True

    # Forward through CNN
    features = model.cnn.conv1(mri_input)
    features = model.cnn.conv2(features)
    features = model.cnn.conv3(features)
    conv_output = model.cnn.conv4(features)

    pooled = model.cnn.pool(conv_output)
    mri_feat = model.cnn.fc(pooled)

    # Dummy forward through other branches
    batch_size = mri_input.shape[0]
    lstm_feat = torch.zeros(batch_size, 64).to(mri_input.device)
    static_feat = torch.zeros(batch_size, 32).to(mri_input.device)

    fused = torch.cat([mri_feat, lstm_feat, static_feat], dim=1)
    output = model.fusion(fused)

    # Backward pass
    model.zero_grad()
    class_score = output[:, target_class].sum()
    class_score.backward()

    # Compute Grad-CAM
    gradients = conv_output.grad
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * conv_output).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam.detach()


def visualize_gradcam_3d(mri, cam, slice_idx=None, save_path=None):
    """
    Visualize 3D Grad-CAM on middle slice
    
    Args:
        mri: MRI tensor (1, 1, D, H, W)
        cam: CAM tensor (1, 1, D, H, W)
        slice_idx: Slice index (default: middle)
        save_path: Path to save figure
    """
    mri_np = mri.cpu().numpy()[0, 0]
    cam_np = cam.cpu().numpy()[0, 0]

    if slice_idx is None:
        slice_idx = mri_np.shape[0] // 2

    # Resize CAM to match MRI
    cam_resized = zoom(
        cam_np,
        [s/c for s, c in zip(mri_np.shape, cam_np.shape)],
        order=1
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original MRI
    axes[0].imshow(mri_np[slice_idx], cmap='gray')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')

    # Grad-CAM
    axes[1].imshow(cam_resized[slice_idx], cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(mri_np[slice_idx], cmap='gray')
    axes[2].imshow(cam_resized[slice_idx], cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_lstm_attention(attention_weights, mmse_scores, save_path=None):
    """
    Visualize LSTM attention over time series
    
    Args:
        attention_weights: Attention matrix (seq_len, seq_len)
        mmse_scores: List of MMSE scores
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 4))

    # Attention heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(attention_weights.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title('LSTM Attention Matrix')
    plt.xlabel('Time Step')
    plt.ylabel('Time Step')

    # Average attention per timestep
    plt.subplot(1, 2, 2)
    attn_mean = attention_weights.mean(dim=0).cpu().numpy()
    plt.bar(range(len(attn_mean)), attn_mean)
    plt.title('Average Attention per Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')

    # Annotate with MMSE scores
    for i, score in enumerate(mmse_scores[:len(attn_mean)]):
        plt.text(i, attn_mean[i], f'{score:.1f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compute_feature_importance(model, val_loader, device):
    """
    Compute static feature importance using Integrated Gradients
    
    Args:
        model: Trained fusion model
        val_loader: Validation DataLoader
        device: torch.device
    
    Returns:
        Mean importance scores (n_features,)
    """
    model.eval()

    # Wrapper for static features only
    def forward_static(static_input):
        batch_size = static_input.shape[0]
        mri_feat = torch.zeros(batch_size, 128).to(device)
        lstm_feat = torch.zeros(batch_size, 64).to(device)
        static_feat = model.static_net(static_input)
        fused = torch.cat([mri_feat, lstm_feat, static_feat], dim=1)
        return model.fusion(fused)

    ig = IntegratedGradients(forward_static)

    all_attributions = []
    for batch in val_loader:
        static_input = batch['cognitive_static'].to(device)
        baseline = torch.zeros_like(static_input)

        attributions = ig.attribute(
            static_input,
            baseline,
            target=batch['label'].to(device)
        )
        all_attributions.append(
            attributions.abs().mean(dim=0).cpu().numpy()
        )

    return np.mean(all_attributions, axis=0)


def plot_feature_importance(importance, feature_names, save_path=None):
    """
    Plot feature importance bar chart
    
    Args:
        importance: Importance scores (n_features,)
        feature_names: List of feature names
        save_path: Path to save figure
    """
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance[sorted_idx])
    plt.yticks(range(len(importance)),
              [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importance Score')
    plt.title('Static Cognitive Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()