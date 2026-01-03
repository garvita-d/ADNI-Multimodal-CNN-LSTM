"""
LSTM Module for Temporal Cognitive Assessment Modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CognitiveLSTM(nn.Module):
    """
    Bidirectional LSTM for modeling cognitive assessment time series
    
    Architecture:
        - 2-layer bidirectional LSTM
        - Attention mechanism (stored for XAI)
        - Fully connected output layer
    
    Args:
        input_size: Input feature dimension (default: 1 for MMSE scores)
        hidden_size: LSTM hidden dimension (default: 64)
        out_features: Output feature dimension (default: 64)
    """
    
    def __init__(self, input_size=1, hidden_size=64, out_features=64):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, out_features)
        self.attention_weights = None  # For XAI visualization

    def forward(self, x):
        """
        Forward pass with attention computation
        
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
        Returns:
            Feature vector (batch_size, out_features)
        """
        # LSTM forward
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Simple attention mechanism
        # Compute pairwise attention scores
        attn_scores = torch.bmm(lstm_out, lstm_out.transpose(1, 2))
        attn_weights = F.softmax(attn_scores.mean(dim=1), dim=1)
        
        # Store for visualization (detached from graph)
        self.attention_weights = attn_weights.detach()
        
        # Concatenate forward and backward hidden states
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        return self.fc(out)


class StaticCogNet(nn.Module):
    """
    MLP for processing static cognitive features
    
    Args:
        input_size: Number of input features
        out_features: Output feature dimension (default: 32)
    """
    
    def __init__(self, input_size, out_features=32):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Static cognitive features (batch_size, input_size)
        Returns:
            Feature vector (batch_size, out_features)
        """
        return self.net(x)