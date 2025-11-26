"""
Helper script to download and organize new training data
Supports multiple data sources
"""

import os
import shutil
from pathlib import Path
import kagglehub
import zipfile
import requests
from tqdm import tqdm


def download_kaggle_dataset(dataset_name, output_dir='data/new_data'):
    """
    Download dataset from Kaggle
    
    Args:
        dataset_name: Kaggle dataset name (e.g., 'anaghachoudhari/pcos-detection-using-ultrasound-images')
        output_dir: Directory to save downloaded data
    """
    print(f"Downloading dataset: {dataset_name}")
    print("Note: You need Kaggle API credentials set up")
    
    try:
        # Download using kagglehub
        path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {path}")
        
        # Copy to output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        if Path(path).is_dir():
            for item in Path(path).iterdir():
                dest = output_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
        
        print(f"Data organized in: {output_dir}")
        return output_path
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nTo use Kaggle datasets, you need to:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Get API credentials from https://www.kaggle.com/settings")
        print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:/Users/<username>/.kaggle/ (Windows)")
        return None


def organize_images_for_training(source_dir, dest_dir='data/train', categories=None):
    """
    Organize downloaded images into training directory structure
    
    Args:
        source_dir: Source directory with images
        dest_dir: Destination training directory
        categories: List of category names (e.g., ['infected', 'notinfected'])
    """
    if categories is None:
        categories = ['infected', 'notinfected']
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Create category directories
    for category in categories:
        (dest_path / category).mkdir(parents=True, exist_ok=True)
    
    # Find and organize images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    organized_count = 0
    
    print(f"Organizing images from {source_dir} to {dest_dir}")
    
    # If source has category subdirectories
    for category_dir in source_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name.lower()
            
            # Map common variations
            category_mapping = {
                'infected': 'infected',
                'pcos': 'infected',
                'positive': 'infected',
                'notinfected': 'notinfected',
                'noninfected': 'notinfected',
                'normal': 'notinfected',
                'negative': 'notinfected',
                'healthy': 'notinfected'
            }
            
            target_category = category_mapping.get(category_name, category_name)
            
            if target_category in categories:
                dest_category_dir = dest_path / target_category
                
                # Copy images
                for img_file in category_dir.rglob('*'):
                    if img_file.suffix.lower() in image_extensions:
                        dest_file = dest_category_dir / img_file.name
                        if not dest_file.exists():
                            shutil.copy2(img_file, dest_file)
                            organized_count += 1
                
                print(f"  Organized {organized_count} images to {target_category}/")
    
    print(f"\nTotal images organized: {organized_count}")
    return organized_count


def validate_and_clean_images(directory):
    """
    Validate images and remove corrupted ones
    
    Args:
        directory: Directory to validate
    """
    from PIL import Image
    from src.preprocessing import find_problematic_images
    
    print(f"Validating images in {directory}...")
    problematic = find_problematic_images(directory)
    
    if problematic:
        print(f"Found {len(problematic)} problematic images:")
        for path, error in problematic[:10]:  # Show first 10
            print(f"  - {path}: {error}")
        
        response = input("\nRemove problematic images? (y/n): ")
        if response.lower() == 'y':
            removed = 0
            for path, _ in problematic:
                try:
                    os.remove(path)
                    removed += 1
                except:
                    pass
            print(f"Removed {removed} problematic images")
    else:
        print("All images are valid!")


def main():
    """Main function to guide user through data download"""
    print("=" * 60)
    print("PCOS Detection - Data Download Helper")
    print("=" * 60)
    print("\nOptions:")
    print("1. Download from Kaggle (requires API setup)")
    print("2. Organize existing images")
    print("3. Validate images")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == '1':
        dataset_name = input("Enter Kaggle dataset name (e.g., 'anaghachoudhari/pcos-detection-using-ultrasound-images'): ")
        output_dir = input("Output directory (default: data/new_data): ") or 'data/new_data'
        download_kaggle_dataset(dataset_name, output_dir)
        
        organize = input("\nOrganize images for training? (y/n): ")
        if organize.lower() == 'y':
            organize_images_for_training(output_dir)
    
    elif choice == '2':
        source_dir = input("Enter source directory path: ")
        dest_dir = input("Enter destination directory (default: data/train): ") or 'data/train'
        organize_images_for_training(source_dir, dest_dir)
    
    elif choice == '3':
        directory = input("Enter directory to validate: ")
        validate_and_clean_images(directory)
    
    else:
        print("Exiting...")


if __name__ == '__main__':
    main()












