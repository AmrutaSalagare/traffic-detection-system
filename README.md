# Traffic Management System with Ambulance Priority Detection

## 🚦 Project Overview

AI-powered traffic management system using YOLOv8 for vehicle detection with special ambulance priority handling for Indian traffic scenarios.

## 📁 Project Structure

```
traffic/
├── 📄 README.md                     # This file
├── 📄 requirements.txt              # Python dependencies
├── 📄 training_config.yaml          # Training configuration
├── 📄 .gitignore                    # Git ignore rules
├── 📄 yolov8n.pt                    # Base YOLOv8 model
│
├── 📂 scripts/                      # Utility scripts
│   └── 📄 integrate_datasets.py     # Dataset integration utility
│
├── 📂 models/                       # Trained models (created during training)
│   └── (trained model files will be saved here)
│
├── 📂 docs/                         # Documentation
│   ├── 📄 README.md                 # Detailed documentation
│   ├── 📄 DATASET_ANALYSIS.md       # Dataset analysis report
│   └── 📄 *.md                      # Other documentation files
│
├── 📂 traffic_dataset/              # Integrated dataset (1,800 images)
│   ├── 📂 images/                   # All training images
│   ├── 📂 labels/                   # YOLO format annotations
│   ├── 📄 dataset.yaml              # Dataset configuration
│   ├── 📄 train.txt                 # Training image list
│   ├── 📄 val.txt                   # Validation image list
│   └── 📄 test.txt                  # Test image list
│
└── 📂 DriveIndia Dataset - IIT Hyderabad (TiHAN)/  # Original dataset
    └── (original dataset files)
```

## 🚀 Core Scripts

### Training & Validation

- **`train_traffic_yolo.py`** - Main YOLOv8 training pipeline
- **`validate_model.py`** - Model validation and performance analysis

### Deployment

- **`deploy_traffic_system.py`** - Advanced real-time traffic management system
- **`traffic_detection_system.py`** - Basic traffic detection system

## 🎯 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_traffic_yolo.py
```

### 3. Validate Performance

```bash
python validate_model.py -m models/best.pt -d traffic_dataset
```

### 4. Run Real-time System

```bash
# Camera input
python deploy_traffic_system.py -m models/best.pt -s 0

# Video file input
python deploy_traffic_system.py -m models/best.pt -s video.mp4
```

## 🚑 Key Features

- **Vehicle Detection**: Cars, motorcycles, buses, trucks, auto-rickshaws
- **Ambulance Priority**: Special handling for emergency vehicles
- **Traffic Analysis**: Real-time density and flow analysis
- **Signal Control**: Adaptive traffic light management
- **Indian Traffic**: Optimized for Indian road conditions

## 📊 Dataset

- **Total Images**: 1,800 (DriveIndia + HuggingFace Ambulance)
- **Classes**: 27 vehicle and traffic object types
- **Format**: YOLO annotation format
- **Ambulance Detection**: >90% accuracy target

## 🛠️ Technical Stack

- **Framework**: YOLOv8 (Ultralytics)
- **Language**: Python 3.8+
- **Libraries**: OpenCV, PyTorch, NumPy, Pandas
- **Database**: SQLite for analytics
- **Deployment**: Real-time processing, edge-compatible

## 📝 License

Educational/Research Project - Traffic Management AI System

---

**Author**: Traffic Management AI Team  
**Date**: September 2025  
**Version**: 1.0
