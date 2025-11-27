"""
Dataset Download Script for Render Build Process
Downloads and organizes PCOS dataset during deployment
"""

import os
import sys
import shutil
from pathlib import Path
import json

# Try to import kagglehub (may not be available, that's OK)
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("⚠️ kagglehub not available - will try alternative download method")

# Dataset configuration
KAGGLE_DATASET = "anaghachoudhari/pcos-detection-using-ultrasound-images"
TEMP_DOWNLOAD_DIR = "data/temp_download"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Create directories
Path(TEMP_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
Path(TEST_DIR).mkdir(parents=True, exist_ok=True)

# Create category subdirectories
for category in ['infected', 'notinfected']:
    (Path(TRAIN_DIR) / category).mkdir(parents=True, exist_ok=True)
    (Path(TEST_DIR) / category).mkdir(parents=True, exist_ok=True)


def download_from_kaggle():
    """Download dataset from Kaggle using kagglehub"""
    if not KAGGLE_AVAILABLE:
        print("❌ kagglehub not available. Install with: pip install kagglehub")
        return None
    
    try:
        print(f"📥 Downloading dataset from Kaggle: {KAGGLE_DATASET}")
        print("   This may take a few minutes...")
        
        # Check for Kaggle credentials
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        # If credentials exist, use them
        if kaggle_json.exists():
            print("   ✅ Kaggle credentials found")
        else:
            # Check environment variables (for Render)
            username = os.getenv('KAGGLE_USERNAME')
            key = os.getenv('KAGGLE_KEY')
            
            if username and key:
                print("   ✅ Kaggle credentials found in environment variables")
                # Create kaggle directory and credentials file
                kaggle_dir.mkdir(exist_ok=True)
                with open(kaggle_json, 'w') as f:
                    json.dump({
                        'username': username,
                        'key': key
                    }, f)
                # Set proper permissions (required by Kaggle)
                os.chmod(kaggle_json, 0o600)
            else:
                print("   ⚠️  No Kaggle credentials found")
                print("   Set KAGGLE_USERNAME and KAGGLE_KEY environment variables on Render")
                return None
        
        # Download dataset
        download_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"✅ Dataset downloaded to: {download_path}")
        
        return Path(download_path)
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        print("   Make sure KAGGLE_USERNAME and KAGGLE_KEY are set in Render environment variables")
        return None


def find_dataset_files(download_path):
    """Find the actual dataset files in the downloaded directory"""
    download_path = Path(download_path)
    
    # Look for common dataset structures
    possible_paths = [
        download_path / "PCOS",
        download_path / "pcos",
        download_path / "dataset",
        download_path / "data",
        download_path
    ]
    
    for path in possible_paths:
        if path.exists():
            # Check if it has train/test or infected/notinfected structure
            if (path / "train").exists() or (path / "infected").exists():
                return path
    
    # If no standard structure, return the download path
    return download_path


def organize_dataset(source_path, train_ratio=0.8):
    """
    Organize dataset into train/test split
    Handles both pre-split and unsplit datasets
    """
    source_path = Path(source_path)
    train_path = Path(TRAIN_DIR)
    test_path = Path(TEST_DIR)
    
    print(f"📁 Organizing dataset from: {source_path}")
    
    # Check if dataset is already split (has train/test directories)
    if (source_path / "train").exists() and (source_path / "test").exists():
        print("   Dataset already has train/test split - copying...")
        
        # Copy train data
        for category in ['infected', 'notinfected', 'noninfected']:
            src_train = source_path / "train" / category
            if not src_train.exists():
                # Try alternative names
                for alt_name in ['pcos', 'positive', 'infected']:
                    if (source_path / "train" / alt_name).exists():
                        src_train = source_path / "train" / alt_name
                        break
                else:
                    continue
            
            # Map to standard names
            if category == 'noninfected':
                dest_category = 'notinfected'
            else:
                dest_category = 'infected' if 'infected' in category.lower() or 'pcos' in category.lower() else 'notinfected'
            
            dest_train = train_path / dest_category
            if src_train.exists():
                print(f"   Copying {src_train.name} → train/{dest_category}")
                shutil.copytree(src_train, dest_train, dirs_exist_ok=True)
        
        # Copy test data
        for category in ['infected', 'notinfected', 'noninfected']:
            src_test = source_path / "test" / category
            if not src_test.exists():
                for alt_name in ['pcos', 'positive', 'infected']:
                    if (source_path / "test" / alt_name).exists():
                        src_test = source_path / "test" / alt_name
                        break
                else:
                    continue
            
            if category == 'noninfected':
                dest_category = 'notinfected'
            else:
                dest_category = 'infected' if 'infected' in category.lower() or 'pcos' in category.lower() else 'notinfected'
            
            dest_test = test_path / dest_category
            if src_test.exists():
                print(f"   Copying {src_test.name} → test/{dest_category}")
                shutil.copytree(src_test, dest_test, dirs_exist_ok=True)
    
    else:
        # Dataset is not split - need to split it
        print(f"   Dataset not pre-split - organizing and splitting {train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test...")
        
        # Find category directories
        categories = {}
        for item in source_path.iterdir():
            if item.is_dir():
                name_lower = item.name.lower()
                if 'infected' in name_lower or 'pcos' in name_lower or 'positive' in name_lower:
                    categories['infected'] = item
                elif 'notinfected' in name_lower or 'noninfected' in name_lower or 'normal' in name_lower or 'negative' in name_lower:
                    categories['notinfected'] = item
        
        # Split each category
        import random
        random.seed(42)  # For reproducibility
        
        for category_name, category_path in categories.items():
            # Get all image files
            image_files = list(category_path.glob('*.jpg')) + \
                         list(category_path.glob('*.jpeg')) + \
                         list(category_path.glob('*.png')) + \
                         list(category_path.glob('*.JPG')) + \
                         list(category_path.glob('*.JPEG')) + \
                         list(category_path.glob('*.PNG'))
            
            # Shuffle and split
            random.shuffle(image_files)
            split_idx = int(len(image_files) * train_ratio)
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]
            
            # Copy to train
            train_dest = train_path / category_name
            for img_file in train_files:
                shutil.copy2(img_file, train_dest / img_file.name)
            
            # Copy to test
            test_dest = test_path / category_name
            for img_file in test_files:
                shutil.copy2(img_file, test_dest / img_file.name)
            
            print(f"   {category_name}: {len(train_files)} train, {len(test_files)} test")


def count_files(directory):
    """Count files in directory"""
    path = Path(directory)
    if not path.exists():
        return 0
    
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        count += len(list(path.rglob(ext)))
    return count


def main():
    """Main function to download and organize dataset"""
    print("=" * 60)
    print("🚀 PCOS Dataset Download for Render Build")
    print("=" * 60)
    print()
    
    # Check if dataset already exists (with reasonable threshold)
    train_count = count_files(TRAIN_DIR)
    test_count = count_files(TEST_DIR)
    min_expected = 100  # Minimum expected images per split
    
    if train_count >= min_expected and test_count >= min_expected:
        print(f"✅ Dataset already exists:")
        print(f"   Train: {train_count} images")
        print(f"   Test: {test_count} images")
        print("   Skipping download (dataset already present)")
        print()
        print("💡 To force re-download, remove data/train and data/test directories")
        return 0
    elif train_count > 0 or test_count > 0:
        print(f"⚠️  Partial dataset found:")
        print(f"   Train: {train_count} images")
        print(f"   Test: {test_count} images")
        print("   Continuing with download to complete dataset...")
        print()
    
    # Download dataset
    download_path = download_from_kaggle()
    
    if download_path is None:
        print("❌ Failed to download dataset")
        print("   Dataset will need to be provided manually or via Git")
        return 1
    
    # Find actual dataset files
    dataset_path = find_dataset_files(download_path)
    print(f"📂 Found dataset at: {dataset_path}")
    
    # Organize dataset
    try:
        organize_dataset(dataset_path)
        
        # Verify organization
        train_count = count_files(TRAIN_DIR)
        test_count = count_files(TEST_DIR)
        
        print()
        print("=" * 60)
        print("✅ Dataset organization complete!")
        print(f"   Train: {train_count} images")
        print(f"   Test: {test_count} images")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error organizing dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

