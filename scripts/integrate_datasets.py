from datasets import load_dataset
from PIL import Image
import os
import shutil

def integrate_datasets():
    print("🚀 Integrating DriveIndia + HuggingFace Ambulance datasets...")
    
    # 1. Load ambulance dataset
    print("📥 Loading ambulance dataset from HuggingFace...")
    ambulance_ds = load_dataset('mlnomad/imnet1k_ambulance')
    
    # 2. Create directory structure
    print("📁 Creating directory structure...")
    os.makedirs('traffic_dataset/images', exist_ok=True)
    os.makedirs('traffic_dataset/annotations', exist_ok=True)
    
    # 3. Copy DriveIndia files
    print("🚗 Processing DriveIndia dataset...")
    driveIndia_path = "DriveIndia Dataset - IIT Hyderabad (TiHAN)"
    driveIndia_count = 0
    
    for file in os.listdir(driveIndia_path):
        if file.endswith('.jpg'):
            shutil.copy(f"{driveIndia_path}/{file}", f"traffic_dataset/images/{file}")
            driveIndia_count += 1
        elif file.endswith('.txt'):
            shutil.copy(f"{driveIndia_path}/{file}", f"traffic_dataset/annotations/{file}")
    
    print(f"✅ Copied {driveIndia_count} DriveIndia images")
    
    # 4. Process ambulance images
    print("🚑 Processing ambulance dataset...")
    ambulance_count = 0
    
    for i, sample in enumerate(ambulance_ds['train']):
        # Save image
        image = sample['image']
        image_path = f'traffic_dataset/images/ambulance_{i+1:04d}.jpg'
        image.save(image_path)
        
        # Create YOLO annotation (full image bounding box for ambulance)
        annotation_path = f'traffic_dataset/annotations/ambulance_{i+1:04d}.txt'
        with open(annotation_path, 'w') as f:
            # Class 26 = ambulance, centered bounding box covering most of image
            f.write("26 0.5 0.5 0.9 0.9\n")
        
        ambulance_count += 1
        
        if i % 100 == 0:
            print(f"  Processed {i+1}/{len(ambulance_ds['train'])} ambulance images...")
    
    print(f"✅ Processed {ambulance_count} ambulance images")
    
    # 5. Create dataset split files
    print("📝 Creating train/val/test splits...")
    
    all_images = []
    for file in os.listdir('traffic_dataset/images'):
        if file.endswith('.jpg'):
            all_images.append(file.replace('.jpg', ''))
    
    # Split: 70% train, 20% val, 10% test
    total = len(all_images)
    train_split = int(0.7 * total)
    val_split = int(0.9 * total)
    
    train_images = all_images[:train_split]
    val_images = all_images[train_split:val_split]
    test_images = all_images[val_split:]
    
    # Write split files
    with open('traffic_dataset/train.txt', 'w') as f:
        for img in train_images:
            f.write(f"images/{img}.jpg\n")
    
    with open('traffic_dataset/val.txt', 'w') as f:
        for img in val_images:
            f.write(f"images/{img}.jpg\n")
    
    with open('traffic_dataset/test.txt', 'w') as f:
        for img in test_images:
            f.write(f"images/{img}.jpg\n")
    
    # 6. Create YOLO config file
    print("⚙️ Creating YOLOv8 configuration...")
    
    config_content = f"""# Traffic Management Dataset Configuration
path: ./traffic_dataset
train: train.txt
val: val.txt
test: test.txt

# Number of classes
nc: 12

# Class names (matching DriveIndia + Ambulance)
names:
  0: car
  1: unknown
  2: motorcycle
  3: bus
  4: unknown
  5: bicycle
  6: auto_rickshaw
  7: truck
  8: van
  9: commercial_vehicle
  10: pedestrian
  11: unknown
  12: unknown
  13: unknown
  14: unknown
  15: unknown
  16: unknown
  17: unknown
  18: unknown
  19: unknown
  20: unknown
  21: unknown
  22: traffic_light
  23: unknown
  24: traffic_sign
  25: unknown
  26: ambulance
"""
    
    with open('traffic_dataset/dataset.yaml', 'w') as f:
        f.write(config_content)
    
    print("\n🎉 DATASET INTEGRATION COMPLETE!")
    print("=" * 50)
    print(f"📊 FINAL STATISTICS:")
    print(f"   Total images: {total}")
    print(f"   DriveIndia traffic images: {driveIndia_count}")
    print(f"   HuggingFace ambulance images: {ambulance_count}")
    print(f"   Train images: {len(train_images)}")
    print(f"   Validation images: {len(val_images)}")
    print(f"   Test images: {len(test_images)}")
    print("=" * 50)
    print("🚀 Ready for YOLOv8 training!")
    print("   Config file: traffic_dataset/dataset.yaml")
    print("   Next step: Run YOLOv8 training command")

if __name__ == "__main__":
    try:
        integrate_datasets()
    except Exception as e:
        print(f"❌ Error during integration: {e}")
        print("Please check dependencies and file paths")
