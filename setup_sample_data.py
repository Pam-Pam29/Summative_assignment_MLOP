"""
Helper script to set up sample data structure for visualizations
Run this script, then copy a few sample images to the directories
"""
from pathlib import Path

# Create directory structure
directories = [
    'data/train/infected',
    'data/train/notinfected',
    'data/test/infected',
    'data/test/notinfected'
]

print("Creating data directory structure...")
for dir_path in directories:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    # Create .gitkeep file
    (Path(dir_path) / '.gitkeep').touch()
    print(f"✅ Created: {dir_path}")

print("\n" + "="*60)
print("Next steps:")
print("="*60)
print("1. Copy 10 sample images from your notebook's training data:")
print("   - Copy to: data/train/infected/")
print("   - Copy to: data/train/notinfected/")
print("\n2. Copy 5 sample images from your notebook's test data:")
print("   - Copy to: data/test/infected/")
print("   - Copy to: data/test/notinfected/")
print("\n3. Verify structure:")
print("   - data/train/infected/ should have ~10 images")
print("   - data/train/notinfected/ should have ~10 images")
print("   - data/test/infected/ should have ~5 images")
print("   - data/test/notinfected/ should have ~5 images")
print("\n4. Commit to git:")
print("   git add data/train/ data/test/")
print("   git commit -m 'Add sample data for visualizations'")
print("   git push origin main")
print("\n✅ Visualizations will work once images are added!")

