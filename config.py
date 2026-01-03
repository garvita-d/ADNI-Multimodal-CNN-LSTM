"""
Configuration file for ADNI Multimodal CNN-LSTM Project
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
FIGURES_DIR = RESULTS_DIR / 'figures'
LOGS_DIR = RESULTS_DIR / 'logs'

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA PATHS - UPDATE THESE TO MATCH YOUR SETUP
# ============================================================================
MRI_DIR = DATA_DIR / 'ADNI'
MMSE_CSV = DATA_DIR / 'MMSE_20Nov2025.csv'
DXSUM_CSV = DATA_DIR / 'DXSUM_20Nov2025.csv'
METADATA_CSV = DATA_DIR / 'ADNI1_Annual_2_Yr_3T_11_20_2025.csv'

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
class ModelConfig:
    # Training
    EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Cross-validation
    N_FOLDS = 5
    
    # Early stopping
    PATIENCE = 15
    
    # Model architecture
    CNN_OUT_FEATURES = 128
    LSTM_HIDDEN_SIZE = 64
    LSTM_OUT_FEATURES = 64
    STATIC_OUT_FEATURES = 32
    NUM_CLASSES = 3
    
    # Input dimensions
    TARGET_SHAPE = (64, 64, 64)  # MRI input size
    MAX_SEQ_LEN = 5  # MMSE time-series length
    
    # Optimization
    GRADIENT_CLIP = 1.0
    
    # Device
    DEVICE = 'cuda'  # or 'cpu'

# ============================================================================
# COGNITIVE FEATURES
# ============================================================================
COG_FEATURES = [
    'MMSCORE', 'MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY',
    'MMSEASON', 'MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE'
]

# ============================================================================
# CLASS LABELS
# ============================================================================
CLASS_NAMES = ['CN', 'MCI', 'AD']
CLASS_MAPPING = {
    1.0: 0,  # CN (Cognitively Normal)
    2.0: 1,  # MCI (Mild Cognitive Impairment)
    3.0: 2   # AD (Alzheimer's Disease)
}

# ============================================================================
# PREPROCESSING
# ============================================================================
class PreprocessConfig:
    # MRI normalization
    Z_SCORE_THRESHOLD = 0.1
    CLIP_MIN = -5
    CLIP_MAX = 5
    
    # MMSE normalization
    MMSE_MAX_SCORE = 30.0

# ============================================================================
# AUGMENTATION
# ============================================================================
class AugmentConfig:
    # TorchIO augmentation parameters
    AFFINE_SCALES = (0.95, 1.05)
    AFFINE_DEGREES = 8
    AFFINE_TRANSLATION = 3
    
    NOISE_STD = (0, 0.03)
    GAMMA_LOG_RANGE = (-0.2, 0.2)

# ============================================================================
# RANDOM SEEDS
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# LOGGING
# ============================================================================
LOG_INTERVAL = 10  # Log every N batches
SAVE_CHECKPOINTS = True
CHECKPOINT_INTERVAL = 5  # Save every N epochs