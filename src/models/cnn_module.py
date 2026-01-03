"""
3D CNN Module for MRI Feature Extraction
"""

import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """
    3D Convolutional Neural Network for MRI spatial feature extraction
    
    Architecture:
        - 4 convolutional blocks (32 -> 64 -> 128 -> 256 filters)
        - Batch normalization + ReLU + MaxPooling
        - Global average pooling
        - Fully connected layers with dropout
    
    Args:
        out_features: Output feature dimension (default: 128)
    """
    
    def __init__(self, out_features=128):
        super().__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_features),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input MRI tensor (batch_size, 1, D, H, W)
        Returns:
            Feature vector (batch_size, out_features)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        return self.fc(x)