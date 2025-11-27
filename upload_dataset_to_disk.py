"""
Helper script to upload dataset to Render Persistent Disk
Run this in Render Shell after persistent disk is set up
"""

import os
import shutil
from pathlib import Path
import sys

# Persistent disk mount path (from render.yaml)
PERSISTENT_DISK_PATH = Path("/opt/render/project/src/data")

# Local dataset paths (adjust if different)
LOCAL_TRAIN_DIR = Path("data/train")
LOCAL_TEST_DIR = Path("data/test")

# Alternative: If dataset is in a different location, specify here
# LOCAL_DATASET_PATH = Path("/path/to/your/dataset")


def check_disk_mount():
    """Check if persistent disk is mounted"""
    if not PERSISTENT_DISK_PATH.exists():
        print(f"❌ Persistent disk not found at: {PERSISTENT_DISK_PATH}")
        print("   Make sure persistent disk is configured in render.yaml")
        return False
    
    # Check if it's writable
    try:
        test_file = PERSISTENT_DISK_PATH / ".test_write"
        test_file.touch()
        test_file.unlink()
        print(f"✅ Persistent disk is mounted and writable: {PERSISTENT_DISK_PATH}")
        return True
    except Exception as e:
        print(f"❌ Persistent disk is not writable: {e}")
        return False


def find_local_dataset():
    """Find local dataset directory"""
    # Try current directory first
    if LOCAL_TRAIN_DIR.exists() and LOCAL_TEST_DIR.exists():
        return Path(".").absolute()
    
    # Try parent directory
    parent = Path("..").absolute()
    if (parent / LOCAL_TRAIN_DIR).exists() and (parent / LOCAL_TEST_DIR).exists():
        return parent
    
    # Try common locations
    common_paths = [
        Path.home() / "data",
        Path("/data"),
        Path("/tmp/dataset"),
    ]
    
    for path in common_paths:
        if (path / "train").exists() and (path / "test").exists():
            return path
    
    return None


def upload_dataset(local_base_path, disk_path):
    """Upload dataset from local to persistent disk"""
    local_base = Path(local_base_path)
    disk = Path(disk_path)
    
    # Create directory structure on disk
    train_disk = disk / "train"
    test_disk = disk / "test"
    
    train_disk.mkdir(parents=True, exist_ok=True)
    test_disk.mkdir(parents=True, exist_ok=True)
    
    # Upload train data
    print("📤 Uploading training data...")
    train_src = local_base / LOCAL_TRAIN_DIR
    if train_src.exists():
        for category_dir in train_src.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                dest_dir = train_disk / category_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy images
                image_files = list(category_dir.glob('*.jpg')) + \
                             list(category_dir.glob('*.jpeg')) + \
                             list(category_dir.glob('*.png')) + \
                             list(category_dir.glob('*.JPG')) + \
                             list(category_dir.glob('*.JPEG')) + \
                             list(category_dir.glob('*.PNG'))
                
                copied = 0
                for img_file in image_files:
                    dest_file = dest_dir / img_file.name
                    if not dest_file.exists():
                        shutil.copy2(img_file, dest_file)
                        copied += 1
                
                print(f"   ✅ {category_name}: {copied} images uploaded")
    else:
        print(f"   ⚠️  Training directory not found: {train_src}")
    
    # Upload test data
    print("📤 Uploading test data...")
    test_src = local_base / LOCAL_TEST_DIR
    if test_src.exists():
        for category_dir in test_src.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                dest_dir = test_disk / category_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy images
                image_files = list(category_dir.glob('*.jpg')) + \
                             list(category_dir.glob('*.jpeg')) + \
                             list(category_dir.glob('*.png')) + \
                             list(category_dir.glob('*.JPG')) + \
                             list(category_dir.glob('*.JPEG')) + \
                             list(category_dir.glob('*.PNG'))
                
                copied = 0
                for img_file in image_files:
                    dest_file = dest_dir / img_file.name
                    if not dest_file.exists():
                        shutil.copy2(img_file, dest_file)
                        copied += 1
                
                print(f"   ✅ {category_name}: {copied} images uploaded")
    else:
        print(f"   ⚠️  Test directory not found: {test_src}")


def verify_upload(disk_path):
    """Verify dataset was uploaded correctly"""
    disk = Path(disk_path)
    
    print("\n🔍 Verifying upload...")
    
    train_dir = disk / "train"
    test_dir = disk / "test"
    
    train_count = 0
    test_count = 0
    
    if train_dir.exists():
        for category_dir in train_dir.iterdir():
            if category_dir.is_dir():
                images = list(category_dir.glob('*.jpg')) + \
                        list(category_dir.glob('*.jpeg')) + \
                        list(category_dir.glob('*.png'))
                count = len(images)
                train_count += count
                print(f"   Train/{category_dir.name}: {count} images")
    
    if test_dir.exists():
        for category_dir in test_dir.iterdir():
            if category_dir.is_dir():
                images = list(category_dir.glob('*.jpg')) + \
                        list(category_dir.glob('*.jpeg')) + \
                        list(category_dir.glob('*.png'))
                count = len(images)
                test_count += count
                print(f"   Test/{category_dir.name}: {count} images")
    
    print(f"\n✅ Total: {train_count} train images, {test_count} test images")
    
    return train_count > 0 and test_count > 0


def main():
    """Main function"""
    print("=" * 60)
    print("📦 Upload Dataset to Render Persistent Disk")
    print("=" * 60)
    print()
    
    # Check if running on Render
    if not os.getenv('RENDER'):
        print("⚠️  This script is designed to run on Render")
        print("   It will still work, but verify paths are correct")
        print()
    
    # Check disk mount
    if not check_disk_mount():
        print("\n❌ Cannot proceed without persistent disk")
        return 1
    
    # Find local dataset
    print("🔍 Looking for local dataset...")
    local_base = find_local_dataset()
    
    if local_base is None:
        print("❌ Could not find local dataset")
        print("\nPlease specify the dataset path:")
        print("   Option 1: Place dataset in 'data/train' and 'data/test'")
        print("   Option 2: Set LOCAL_DATASET_PATH in this script")
        print("   Option 3: Use scp/rsync to upload directly")
        return 1
    
    print(f"✅ Found dataset at: {local_base}")
    print(f"   Train: {local_base / LOCAL_TRAIN_DIR}")
    print(f"   Test: {local_base / LOCAL_TEST_DIR}")
    print()
    
    # Confirm upload
    response = input("Proceed with upload? (y/n): ")
    if response.lower() != 'y':
        print("Upload cancelled")
        return 0
    
    # Upload dataset
    try:
        upload_dataset(local_base, PERSISTENT_DISK_PATH)
        
        # Verify
        if verify_upload(PERSISTENT_DISK_PATH):
            print("\n" + "=" * 60)
            print("✅ Dataset upload complete!")
            print("=" * 60)
            print("\n💡 Next steps:")
            print("   1. Restart your Render service")
            print("   2. Check /dataset_stats endpoint")
            print("   3. Verify dataset is accessible")
            return 0
        else:
            print("\n⚠️  Upload completed but verification failed")
            print("   Check the persistent disk manually")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

