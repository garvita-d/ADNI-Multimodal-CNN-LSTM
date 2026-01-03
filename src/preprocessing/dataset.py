"""
PyTorch Dataset for ADNI Multimodal Data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torchio as tio
from .data_preprocessing import preprocess_mri


class ADNIDataset(Dataset):
    """
    PyTorch Dataset for multimodal ADNI data
    
    Args:
        records: List of dataset records (from build_dataset_records)
        transform: TorchIO transforms for data augmentation
        max_seq_len: Maximum length of MMSE time series (default: 5)
        preload: If True, preload all MRI data into memory (default: False)
    """
    
    def __init__(self, records, transform=None, max_seq_len=5, preload=False):
        self.records = records
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.preload = preload

        if preload:
            print("Preloading MRI data...")
            for r in tqdm(records):
                r['mri_data'] = preprocess_mri(r['mri_path'])

    def __len__(self):
        return len(self.records)

    def prepare_mmse_sequence(self, scores):
        """
        Prepare MMSE time series with zero-padding
        
        Args:
            scores: List of MMSE scores
        Returns:
            Normalized and padded numpy array (max_seq_len, 1)
        """
        # Normalize scores to [0, 1]
        scores = [s / 30.0 for s in scores]
        
        # Pad or truncate to max_seq_len
        if len(scores) < self.max_seq_len:
            scores.extend([0.0] * (self.max_seq_len - len(scores)))
        else:
            scores = scores[:self.max_seq_len]
        
        return np.array(scores, dtype=np.float32).reshape(-1, 1)

    def __getitem__(self, idx):
        """
        Get a single data sample
        
        Returns:
            Dictionary with keys:
                - 'mri': Tensor (1, D, H, W)
                - 'cognitive_seq': Tensor (max_seq_len, 1)
                - 'cognitive_static': Tensor (n_features,)
                - 'label': Tensor (scalar)
                - 'subject_id': String
        """
        r = self.records[idx]

        # Load MRI
        if self.preload:
            mri = r['mri_data']
        else:
            mri = preprocess_mri(r['mri_path'])

        mri_tensor = torch.from_numpy(mri).unsqueeze(0)  # Add channel: (1, D, H, W)

        # Apply augmentation if provided
        if self.transform:
            subject = tio.Subject(mri=tio.ScalarImage(tensor=mri_tensor))
            transformed = self.transform(subject)
            mri_tensor = transformed['mri'].data

        # Prepare cognitive data
        cog_seq = self.prepare_mmse_sequence(r['mmse_series'])
        cog_tensor = torch.from_numpy(cog_seq)
        static_cog = torch.from_numpy(r['cognitive'])
        label = torch.tensor(r['label'], dtype=torch.long)

        return {
            'mri': mri_tensor,
            'cognitive_seq': cog_tensor,
            'cognitive_static': static_cog,
            'label': label,
            'subject_id': r['subject_id']
        }


def get_train_transforms():
    """
    Get training data augmentation transforms
    
    Returns:
        TorchIO Compose transform
    """
    return tio.Compose([
        tio.RandomAffine(scales=(0.95, 1.05), degrees=8, translation=3),
        tio.RandomNoise(std=(0, 0.03)),
        tio.RandomGamma(log_gamma=(-0.2, 0.2)),
    ])