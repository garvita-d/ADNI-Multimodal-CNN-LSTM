"""
Data Preprocessing Module
Handles MRI loading, cognitive data processing, and dataset creation
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm


TARGET_SHAPE = (64, 64, 64)  # MRI target dimensions


def find_nifti(folder_path):
    """
    Find first NIfTI file in folder recursively
    
    Args:
        folder_path: Path to search
    Returns:
        Path to .nii or .nii.gz file, or None
    """
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(('.nii', '.nii.gz')):
                return os.path.join(root, f)
    return None


def extract_rid(folder_name):
    """
    Extract RID (subject ID) from ADNI folder name
    Example: '002_S_1018' -> 1018
    
    Args:
        folder_name: ADNI folder name
    Returns:
        Integer RID or None
    """
    parts = folder_name.split('_')
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except:
            pass
    return None


def preprocess_mri(nifti_path, target_shape=TARGET_SHAPE):
    """
    Load, resize, and normalize MRI scan
    
    Args:
        nifti_path: Path to .nii/.nii.gz file
        target_shape: Desired output shape (D, H, W)
    Returns:
        Preprocessed 3D numpy array
    """
    try:
        # Load NIfTI
        img = nib.load(nifti_path)
        data = img.get_fdata().astype(np.float32)

        # Resize to target shape
        factors = [t/s for t, s in zip(target_shape, data.shape)]
        data = zoom(data, factors, order=1)

        # Z-score normalization (brain tissue only)
        mask = data > data.mean() * 0.1
        if mask.sum() > 0:
            mean, std = data[mask].mean(), data[mask].std()
            data = (data - mean) / (std + 1e-8)

        # Clip outliers
        data = np.clip(data, -5, 5)
        
        return data
    except Exception as e:
        print(f"Error loading {nifti_path}: {e}")
        return np.zeros(target_shape, dtype=np.float32)


def load_cognitive_data(mmse_csv, dxsum_csv):
    """
    Load and merge MMSE and diagnosis data
    
    Args:
        mmse_csv: Path to MMSE CSV
        dxsum_csv: Path to DXSUM CSV
    Returns:
        Merged DataFrame with cognitive scores and diagnosis
    """
    mmse_df = pd.read_csv(mmse_csv)
    dxsum_df = pd.read_csv(dxsum_csv)

    # Get baseline data per subject
    dxsum_bl = dxsum_df.groupby('RID').first().reset_index()
    mmse_bl = mmse_df.groupby('RID').first().reset_index()

    # Merge on RID
    merged_df = mmse_bl.merge(
        dxsum_bl[['RID', 'DIAGNOSIS']], 
        on='RID', 
        how='inner'
    )
    
    return merged_df, mmse_df


def build_dataset_records(mri_dir, merged_df, mmse_df, cog_features):
    """
    Match MRI scans with cognitive data and create dataset records
    
    Args:
        mri_dir: Directory containing MRI folders
        merged_df: Merged cognitive data (baseline)
        mmse_df: Full MMSE data (all visits)
        cog_features: List of cognitive feature column names
    Returns:
        List of dataset records (dicts)
    """
    dataset_records = []

    for folder in tqdm(os.listdir(mri_dir), desc="Matching MRI with metadata"):
        folder_path = os.path.join(mri_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Extract subject ID
        rid = extract_rid(folder)
        if rid is None:
            continue

        # Find subject in metadata
        subj = merged_df[merged_df['RID'] == rid]
        if len(subj) == 0:
            continue

        # Find MRI file
        nifti_path = find_nifti(folder_path)
        if nifti_path is None:
            continue

        # Get diagnosis label
        dx = subj['DIAGNOSIS'].iloc[0]
        if dx == 1.0:
            label, label_str = 0, 'CN'
        elif dx == 2.0:
            label, label_str = 1, 'MCI'
        elif dx == 3.0:
            label, label_str = 2, 'AD'
        else:
            continue

        # Get static cognitive features
        cog_data = subj[cog_features].iloc[0].values.astype(np.float32)
        cog_data = np.nan_to_num(cog_data, 0)

        # Get MMSE time series (multiple visits)
        subj_mmse = mmse_df[mmse_df['RID'] == rid].sort_values('VISCODE')
        mmse_scores = subj_mmse['MMSCORE'].dropna().tolist()
        if len(mmse_scores) == 0:
            mmse_scores = [0.0]

        # Store record
        dataset_records.append({
            'subject_id': folder,
            'rid': rid,
            'mri_path': nifti_path,
            'label': label,
            'label_str': label_str,
            'cognitive': cog_data,
            'mmse_series': mmse_scores
        })

    return dataset_records


def get_cognitive_features(merged_df):
    """
    Get list of available cognitive features from dataframe
    
    Args:
        merged_df: Merged cognitive DataFrame
    Returns:
        List of feature column names
    """
    cog_features = [
        'MMSCORE', 'MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY',
        'MMSEASON', 'MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE'
    ]
    return [c for c in cog_features if c in merged_df.columns]