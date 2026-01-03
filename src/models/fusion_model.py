"""
Multimodal Fusion Model
Combines CNN, LSTM, and static cognitive features
"""

import torch
import torch.nn as nn
from .cnn_module import CNN3D
from .lstm_module import CognitiveLSTM, StaticCogNet


class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion model for Alzheimer's Disease classification
    
    Combines three input modalities:
        1. MRI scans (via 3D CNN)
        2. Cognitive time series (via Bi-LSTM)
        3. Static cognitive features (via MLP)
    
    Architecture:
        - CNN: 128-dim features
        - LSTM: 64-dim features
        - Static: 32-dim features
        - Fusion: 224 -> 128 -> 64 -> 3 classes
    
    Args:
        static_cog_size: Number of static cognitive features
        num_classes: Number of output classes (default: 3 for CN/MCI/AD)
    """
    
    def __init__(self, static_cog_size, num_classes=3):
        super().__init__()
        
        # Individual modality encoders
        self.cnn = CNN3D(out_features=128)
        self.lstm = CognitiveLSTM(input_size=1, hidden_size=64, out_features=64)
        self.static_net = StaticCogNet(static_cog_size, out_features=32)
        
        # Fusion classifier
        # Total: 128 (CNN) + 64 (LSTM) + 32 (Static) = 224
        self.fusion = nn.Sequential(
            nn.Linear(224, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, mri, cog_seq, cog_static):
        """
        Forward pass
        
        Args:
            mri: MRI tensor (batch_size, 1, D, H, W)
            cog_seq: Cognitive time series (batch_size, seq_len, 1)
            cog_static: Static cognitive features (batch_size, n_features)
        
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Extract features from each modality
        mri_feat = self.cnn(mri)
        lstm_feat = self.lstm(cog_seq)
        static_feat = self.static_net(cog_static)
        
        # Concatenate features
        fused = torch.cat([mri_feat, lstm_feat, static_feat], dim=1)
        
        # Classification
        return self.fusion(fused)