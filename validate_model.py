"""
Model Testing and Validation Suite
=================================

This script provides comprehensive testing and validation for the trained
traffic management model with focus on ambulance detection performance.

Features:
- Model validation on test set
- Ambulance detection performance analysis
- Confusion matrix generation
- Real-time performance benchmarking
- Model comparison utilities

Author: Traffic Management System Team
Date: September 2025
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import time
from collections import defaultdict
import argparse

class ModelValidator:
    def __init__(self, model_path, dataset_path):
        """Initialize model validator."""
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        
        # Load model
        print(f"🔄 Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✅ Model loaded successfully!")
        
        # Class names
        self.class_names = {
            0: 'car', 2: 'motorcycle', 3: 'bus', 5: 'bicycle',
            6: 'auto_rickshaw', 7: 'truck', 8: 'van', 9: 'commercial_vehicle',
            10: 'pedestrian', 22: 'traffic_light', 24: 'traffic_sign',
            26: 'ambulance'
        }
        
        self.results = {}
    
    def validate_on_test_set(self):
        """Validate model on the test dataset."""
        print("\n🧪 Running validation on test set...")
        
        # Run validation
        results = self.model.val(
            data=str(self.dataset_path / "dataset.yaml"),
            split='test',
            imgsz=640,
            conf=0.001,
            iou=0.6,
            max_det=300,
            save_json=True,
            save_hybrid=True,
            plots=True,
            verbose=True
        )
        
        # Extract key metrics
        self.results['overall_map50'] = float(results.box.map50)
        self.results['overall_map'] = float(results.box.map)
        self.results['precision'] = float(results.box.mp)
        self.results['recall'] = float(results.box.mr)
        
        # Per-class metrics
        if hasattr(results.box, 'maps'):
            self.results['per_class_map'] = {}
            for i, class_map in enumerate(results.box.maps):
                if i in self.class_names:
                    self.results['per_class_map'][self.class_names[i]] = float(class_map)
        
        print("✅ Validation completed!")
        return results
    
    def test_ambulance_detection(self, ambulance_images_dir=None):
        """Specific testing for ambulance detection performance."""
        print("\n🚑 Testing ambulance detection performance...")
        
        if ambulance_images_dir is None:
            # Use ambulance images from dataset
            images_dir = self.dataset_path / "images"
            ambulance_images = list(images_dir.glob("ambulance_*.jpg"))
        else:
            ambulance_images = list(Path(ambulance_images_dir).glob("*.jpg"))
        
        if not ambulance_images:
            print("❌ No ambulance images found for testing!")
            return {}
        
        print(f"📊 Testing on {len(ambulance_images)} ambulance images...")
        
        ambulance_results = {
            'total_images': len(ambulance_images),
            'detected': 0,
            'missed': 0,
            'false_positives': 0,
            'confidences': [],
            'detection_times': []
        }
        
        for img_path in ambulance_images[:50]:  # Test first 50 for speed
            start_time = time.time()
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Run detection
            results = self.model(image, conf=0.1, verbose=False)
            
            detection_time = time.time() - start_time
            ambulance_results['detection_times'].append(detection_time)
            
            # Check for ambulance detection
            ambulance_detected = False
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        if class_id == 26:  # Ambulance class
                            ambulance_detected = True
                            ambulance_results['confidences'].append(confidence)
                            break
            
            if ambulance_detected:
                ambulance_results['detected'] += 1
            else:
                ambulance_results['missed'] += 1
        
        # Calculate metrics
        total_tested = ambulance_results['detected'] + ambulance_results['missed']
        ambulance_results['detection_rate'] = (
            ambulance_results['detected'] / total_tested if total_tested > 0 else 0
        )
        ambulance_results['avg_confidence'] = (
            np.mean(ambulance_results['confidences']) 
            if ambulance_results['confidences'] else 0
        )
        ambulance_results['avg_detection_time'] = np.mean(ambulance_results['detection_times'])
        
        print(f"✅ Ambulance detection rate: {ambulance_results['detection_rate']:.2%}")
        print(f"📊 Average confidence: {ambulance_results['avg_confidence']:.3f}")
        print(f"⏱️  Average detection time: {ambulance_results['avg_detection_time']:.3f}s")
        
        self.results['ambulance_performance'] = ambulance_results
        return ambulance_results
    
    def benchmark_performance(self, test_images=100):
        """Benchmark model performance on various metrics."""
        print(f"\n⚡ Benchmarking performance on {test_images} images...")
        
        # Get test images
        images_dir = self.dataset_path / "images"
        test_image_paths = list(images_dir.glob("*.jpg"))[:test_images]
        
        benchmark_results = {
            'total_images': len(test_image_paths),
            'total_detections': 0,
            'processing_times': [],
            'fps_values': [],
            'memory_usage': [],
            'class_detections': defaultdict(int)
        }
        
        print("🔄 Processing images...")
        for i, img_path in enumerate(test_image_paths):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(test_image_paths)}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Measure processing time
            start_time = time.time()
            results = self.model(image, conf=0.5, verbose=False)
            processing_time = time.time() - start_time
            
            benchmark_results['processing_times'].append(processing_time)
            benchmark_results['fps_values'].append(1.0 / processing_time)
            
            # Count detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        if class_id in self.class_names:
                            class_name = self.class_names[class_id]
                            benchmark_results['class_detections'][class_name] += 1
                            benchmark_results['total_detections'] += 1
        
        # Calculate summary statistics
        benchmark_results['avg_processing_time'] = np.mean(benchmark_results['processing_times'])
        benchmark_results['avg_fps'] = np.mean(benchmark_results['fps_values'])
        benchmark_results['min_fps'] = np.min(benchmark_results['fps_values'])
        benchmark_results['max_fps'] = np.max(benchmark_results['fps_values'])
        
        print("✅ Benchmarking completed!")
        print(f"📊 Average FPS: {benchmark_results['avg_fps']:.1f}")
        print(f"⏱️  Average processing time: {benchmark_results['avg_processing_time']:.3f}s")
        print(f"🎯 Total detections: {benchmark_results['total_detections']}")
        
        self.results['benchmark'] = benchmark_results
        return benchmark_results
    
    def generate_confusion_matrix(self, sample_size=200):
        """Generate confusion matrix for model predictions."""
        print(f"\n📊 Generating confusion matrix (sample size: {sample_size})...")
        
        # This is a simplified version - in practice you'd need ground truth labels
        # For now, we'll analyze prediction confidence distribution
        
        images_dir = self.dataset_path / "images"
        test_images = list(images_dir.glob("*.jpg"))[:sample_size]
        
        predictions = defaultdict(list)
        
        for img_path in test_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            results = self.model(image, conf=0.3, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        if class_id in self.class_names:
                            class_name = self.class_names[class_id]
                            predictions[class_name].append(confidence)
        
        # Create confidence distribution plot
        plt.figure(figsize=(12, 8))
        
        # Plot confidence distributions
        for class_name, confidences in predictions.items():
            if confidences:  # Only plot classes with detections
                plt.hist(confidences, alpha=0.6, label=f"{class_name} ({len(confidences)})",
                        bins=20, density=True)
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.title('Detection Confidence Distribution by Class')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Confidence distribution plot saved as 'confidence_distribution.png'")
        
        self.results['confidence_distribution'] = dict(predictions)
        return predictions
    
    def test_real_time_performance(self, duration=30):
        """Test real-time performance using webcam."""
        print(f"\n📹 Testing real-time performance for {duration} seconds...")
        print("Note: This requires a camera. Press 'q' to stop early.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not access camera for real-time testing")
            return {}
        
        start_time = time.time()
        frame_count = 0
        processing_times = []
        detection_counts = []
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                proc_start = time.time()
                results = self.model(frame, conf=0.5, verbose=False)
                proc_time = time.time() - proc_start
                processing_times.append(proc_time)
                
                # Count detections
                detection_count = 0
                ambulance_detected = False
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        detection_count += len(boxes)
                        for box in boxes:
                            class_id = int(box.cls[0].cpu().numpy())
                            if class_id == 26:  # Ambulance
                                ambulance_detected = True
                
                detection_counts.append(detection_count)
                
                # Draw simple visualization
                if ambulance_detected:
                    cv2.putText(frame, "AMBULANCE DETECTED!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                fps = 1.0 / proc_time if proc_time > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Real-time Performance Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Test interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Calculate results
        total_time = time.time() - start_time
        real_time_results = {
            'duration': total_time,
            'frames_processed': frame_count,
            'avg_fps': frame_count / total_time,
            'avg_processing_time': np.mean(processing_times),
            'avg_detections_per_frame': np.mean(detection_counts),
            'total_detections': sum(detection_counts)
        }
        
        print("✅ Real-time performance test completed!")
        print(f"📊 Average FPS: {real_time_results['avg_fps']:.1f}")
        print(f"⏱️  Average processing time: {real_time_results['avg_processing_time']:.3f}s")
        
        self.results['real_time_performance'] = real_time_results
        return real_time_results
    
    def generate_report(self, output_path="validation_report.json"):
        """Generate comprehensive validation report."""
        print(f"\n📋 Generating validation report...")
        
        report = {
            'model_path': str(self.model_path),
            'dataset_path': str(self.dataset_path),
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results
        }
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("TRAFFIC MANAGEMENT SYSTEM - MODEL VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            if 'overall_map50' in self.results:
                f.write(f"Overall Performance:\n")
                f.write(f"  mAP@0.5: {self.results['overall_map50']:.3f}\n")
                f.write(f"  mAP@0.5:0.95: {self.results['overall_map']:.3f}\n")
                f.write(f"  Precision: {self.results['precision']:.3f}\n")
                f.write(f"  Recall: {self.results['recall']:.3f}\n\n")
            
            if 'ambulance_performance' in self.results:
                amb = self.results['ambulance_performance']
                f.write(f"Ambulance Detection Performance:\n")
                f.write(f"  Detection Rate: {amb['detection_rate']:.2%}\n")
                f.write(f"  Average Confidence: {amb['avg_confidence']:.3f}\n")
                f.write(f"  Average Detection Time: {amb['avg_detection_time']:.3f}s\n\n")
            
            if 'benchmark' in self.results:
                bench = self.results['benchmark']
                f.write(f"Performance Benchmark:\n")
                f.write(f"  Average FPS: {bench['avg_fps']:.1f}\n")
                f.write(f"  Processing Time: {bench['avg_processing_time']:.3f}s\n")
                f.write(f"  Total Detections: {bench['total_detections']}\n\n")
            
            f.write("Class Detection Summary:\n")
            if 'benchmark' in self.results:
                for class_name, count in self.results['benchmark']['class_detections'].items():
                    f.write(f"  {class_name}: {count}\n")
        
        print(f"✅ Validation report saved:")
        print(f"  📄 JSON report: {output_path}")
        print(f"  📝 Summary: {summary_path}")
        
        return report

def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description='Model Validation Suite')
    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--dataset', '-d', required=True,
                       help='Path to dataset directory')
    parser.add_argument('--ambulance-dir', '-a',
                       help='Directory with additional ambulance test images')
    parser.add_argument('--output', '-o', default='validation_report.json',
                       help='Output report path')
    parser.add_argument('--skip-realtime', action='store_true',
                       help='Skip real-time camera test')
    
    args = parser.parse_args()
    
    print("🧪 Traffic Management System - Model Validation Suite")
    print("=" * 60)
    
    try:
        # Initialize validator
        validator = ModelValidator(args.model, args.dataset)
        
        # Run validation tests
        print("\n1. Validating on test dataset...")
        validator.validate_on_test_set()
        
        print("\n2. Testing ambulance detection...")
        validator.test_ambulance_detection(args.ambulance_dir)
        
        print("\n3. Benchmarking performance...")
        validator.benchmark_performance()
        
        print("\n4. Generating confusion matrix...")
        validator.generate_confusion_matrix()
        
        if not args.skip_realtime:
            print("\n5. Testing real-time performance...")
            validator.test_real_time_performance()
        
        # Generate final report
        print("\n6. Generating validation report...")
        report = validator.generate_report(args.output)
        
        print("\n🎉 VALIDATION COMPLETE!")
        print("=" * 60)
        print(f"📊 Overall mAP@0.5: {report['results'].get('overall_map50', 'N/A')}")
        if 'ambulance_performance' in report['results']:
            amb_rate = report['results']['ambulance_performance']['detection_rate']
            print(f"🚑 Ambulance Detection Rate: {amb_rate:.2%}")
        if 'benchmark' in report['results']:
            fps = report['results']['benchmark']['avg_fps']
            print(f"⚡ Average FPS: {fps:.1f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
