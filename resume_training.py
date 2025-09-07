#!/usr/bin/env python3
"""
Resume Training Script - Continue from last checkpoint with GPU
"""
from ultralytics import YOLO
import torch


def resume_training():
    """Resume training from the last checkpoint with GPU acceleration."""

    print("🔄 Resuming Training with GPU")
    print("=" * 50)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        device = 0  # Use GPU
    else:
        print("⚠️  No GPU found, using CPU")
        device = 'cpu'

    # Path to the last checkpoint
    checkpoint_path = "runs/traffic_yolo_20250907_002542/weights/last.pt"

    try:
        # Load the model from checkpoint
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)

        # Resume training with GPU
        print("🚀 Resuming training...")
        results = model.train(
            resume=True,  # This will resume from the checkpoint
            device=device  # Force GPU usage
        )

        print("✅ Training completed successfully!")
        return results

    except Exception as e:
        print(f"❌ Error resuming training: {e}")
        return None


if __name__ == "__main__":
    resume_training()
