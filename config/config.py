import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = BASE_DIR / "data"
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
    SAMPLES_DIR = BASE_DIR / "samples"
    
    # Create directories if they don't exist
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataset parameters
    IAM_WORDS_DIR = DATA_DIR / "iam" / "words"
    IAM_ASCII_FILE = DATA_DIR / "iam" / "ascii" / "words.txt"
    
    # Model parameters
    IMG_HEIGHT = 32
    IMG_WIDTH = 128
    CHANNELS = 1
    NOISE_DIM = 100
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5
    BETA_2 = 0.999
    
    # Visualization
    SAMPLE_INTERVAL = 10
    NUM_EXAMPLES = 10
    
    # Metrics
    METRICS_DIR = BASE_DIR / "metrics"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
