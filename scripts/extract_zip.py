"""
Utility script to extract ADNI zip files
"""

import os
import zipfile
import sys


def extract_adni_zips(zip_paths, output_dir):
    """
    Extract multiple ADNI zip files
    
    Args:
        zip_paths: List of paths to zip files
        output_dir: Directory to extract to
    """
    print("="*60)
    print("EXTRACTING ZIP FILES")
    print("="*60)
    
    for zip_path in zip_paths:
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                    print(f"✓ Extracted: {zip_path}")
            except Exception as e:
                print(f"✗ Error extracting {zip_path}: {e}")
        else:
            print(f"✗ Not found: {zip_path}")


if __name__ == "__main__":
    # Example usage
    zip_files = [
        'data/ADNI1_Annual 2 Yr 3T.zip',
        'data/ADNI1_Annual 2 Yr 3T1.zip'
    ]
    
    output_directory = 'data/'
    
    extract_adni_zips(zip_files, output_directory)
    
    print("\nDone! Check the 'data/' folder for extracted files.")