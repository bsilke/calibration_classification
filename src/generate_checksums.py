"""
Generate checksums for raw data files.
Run this script to create .sha256 files for data integrity verification.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reproducibility import setup_logging, save_checksum, verify_checksum


def main():
    # Setup logging
    logger = setup_logging('logs/checksum.log')
    
    # Files to checksum
    raw_data_files = [
        'data/raw/net_zero_tracker.csv',
        'data/raw/metadata_glossary.csv',
    ]
    
    interim_files = [
        'data/interim/train.csv',
        'data/interim/val.csv',
        'data/interim/test.csv',
    ]
    
    print("=" * 70)
    print("GENERATING CHECKSUMS FOR DATA FILES")
    print("=" * 70)
    
    # Generate checksums for raw data
    print("\n📁 Raw Data Files:")
    for file_path in raw_data_files:
        if os.path.exists(file_path):
            checksum = save_checksum(file_path)
            print(f"   ✅ {file_path}")
            print(f"      SHA256: {checksum[:32]}...")
        else:
            print(f"   ⚠️ {file_path} - NOT FOUND")
    
    # Generate checksums for interim data
    print("\n📁 Interim Data Files:")
    for file_path in interim_files:
        if os.path.exists(file_path):
            checksum = save_checksum(file_path)
            print(f"   ✅ {file_path}")
            print(f"      SHA256: {checksum[:32]}...")
        else:
            print(f"   ⚠️ {file_path} - NOT FOUND")
    
    print("\n" + "=" * 70)
    print("CHECKSUM GENERATION COMPLETE")
    print("=" * 70)
    print("\nChecksum files created with .sha256 extension")
    print("Use these to verify data integrity before running pipeline")


if __name__ == '__main__':
    main()
