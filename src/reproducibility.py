"""
Reproducibility utilities for the ML pipeline.
Handles logging, random seeds, checksums, and configuration management.
"""

import logging
import hashlib
import json
import random
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def setup_logging(log_file: str = 'logs/preprocess.log', level=logging.INFO):
    """
    Set up logging to both file and console.
    
    Args:
        log_file: Path to log file
        level: Logging level
    
    Returns:
        Logger instance
    """
    # Create logs directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info(f"LOGGING SESSION STARTED: {datetime.now().isoformat()}")
    logger.info("=" * 70)
    
    return logger


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set sklearn seed if available
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass
    
    logger = logging.getLogger(__name__)
    logger.info(f"Random seeds set to {seed} (random, numpy, sklearn)")


def generate_checksum(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Generate a checksum for a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('sha256' or 'md5')
    
    Returns:
        Checksum string
    """
    hash_func = hashlib.sha256() if algorithm == 'sha256' else hashlib.md5()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def save_checksum(file_path: str, output_path: str = None, algorithm: str = 'sha256'):
    """
    Generate and save checksum for a file.
    
    Args:
        file_path: Path to the file to checksum
        output_path: Path to save checksum (default: file_path + .sha256)
        algorithm: Hash algorithm
    
    Returns:
        Checksum string
    """
    checksum = generate_checksum(file_path, algorithm)
    
    if output_path is None:
        output_path = f"{file_path}.{algorithm}"
    
    with open(output_path, 'w') as f:
        f.write(f"{checksum}  {os.path.basename(file_path)}\n")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Checksum ({algorithm}) saved: {output_path}")
    logger.info(f"  Value: {checksum[:16]}...{checksum[-16:]}")
    
    return checksum


def verify_checksum(file_path: str, checksum_path: str = None, algorithm: str = 'sha256') -> bool:
    """
    Verify a file's checksum against a stored value.
    
    Args:
        file_path: Path to the file to verify
        checksum_path: Path to checksum file (default: file_path + .sha256)
        algorithm: Hash algorithm
    
    Returns:
        True if checksum matches, False otherwise
    """
    if checksum_path is None:
        checksum_path = f"{file_path}.{algorithm}"
    
    if not os.path.exists(checksum_path):
        logger = logging.getLogger(__name__)
        logger.warning(f"Checksum file not found: {checksum_path}")
        return False
    
    with open(checksum_path, 'r') as f:
        stored_checksum = f.read().split()[0]
    
    current_checksum = generate_checksum(file_path, algorithm)
    
    logger = logging.getLogger(__name__)
    if current_checksum == stored_checksum:
        logger.info(f"✅ Checksum verified: {os.path.basename(file_path)}")
        return True
    else:
        logger.error(f"❌ Checksum MISMATCH: {os.path.basename(file_path)}")
        logger.error(f"   Expected: {stored_checksum[:32]}...")
        logger.error(f"   Got:      {current_checksum[:32]}...")
        return False


def load_config(config_path: str = 'RUN_CONFIG.json') -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration loaded from: {config_path}")
    
    return config


def save_config(config: dict, config_path: str = 'RUN_CONFIG.json'):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    config['last_updated'] = datetime.now().strftime('%Y-%m-%d')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration saved to: {config_path}")


def log_dataframe_info(df, name: str = 'DataFrame'):
    """
    Log summary information about a DataFrame.
    
    Args:
        df: pandas DataFrame
        name: Name for logging
    """
    logger = logging.getLogger(__name__)
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} columns: {df.columns.tolist()}")
    logger.info(f"{name} missing values: {df.isnull().sum().sum()}")
    logger.info(f"{name} dtypes: {df.dtypes.value_counts().to_dict()}")


def log_transformation(operation: str, before_shape: tuple, after_shape: tuple, 
                       details: str = None):
    """
    Log a data transformation step.
    
    Args:
        operation: Name of the transformation
        before_shape: Shape before transformation
        after_shape: Shape after transformation
        details: Additional details
    """
    logger = logging.getLogger(__name__)
    logger.info(f"TRANSFORM: {operation}")
    logger.info(f"  Before: {before_shape} → After: {after_shape}")
    if details:
        logger.info(f"  Details: {details}")


class PipelineLogger:
    """Context manager for logging pipeline steps."""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"▶ STARTING: {self.step_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"✓ COMPLETED: {self.step_name} ({duration:.2f}s)")
        else:
            self.logger.error(f"✗ FAILED: {self.step_name} - {exc_val}")
        return False


if __name__ == '__main__':
    # Test the utilities
    setup_logging('logs/test.log')
    set_random_seeds(42)
    
    print("Reproducibility utilities loaded successfully!")
