"""
Real-time Traffic Management and Ambulance Detection System
==========================================================

This script provides real-time vehicle detection and ambulance priority system
using the trained YOLOv8 model. Optimized for Indian traffic conditions.

Features:
- Real-time vehicle detection
- Ambulance priority detection
- Traffic density analysis
- Signal control interface
- Performance monitoring

Author: Traffic Management System Team
Date: September 2025
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from pathlib import Path
import json
import threading
from collections import defaultdict, deque
from datetime import datetime
import argparse

class TrafficManagementSystem:
    def __init__(self, model_path, config_path="config.json"):
        """Initialize the Traffic Management System."""
        self.model_path = Path(model_path)
        self.config = self.load_config(config_path)
        
        # Load trained model
        print(f"🔄 Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✅ Model loaded successfully!")
        
        # Class names mapping
        self.class_names = {
            0: 'car', 2: 'motorcycle', 3: 'bus', 5: 'bicycle',
            6: 'auto_rickshaw', 7: 'truck', 8: 'van', 9: 'commercial_vehicle',
            10: 'pedestrian', 22: 'traffic_light', 24: 'traffic_sign',
            26: 'ambulance'  # Priority class
        }
        
        # Colors for visualization
        self.colors = {
            'car': (0, 255, 0), 'motorcycle': (255, 0, 0), 'bus': (0, 0, 255),
            'bicycle': (255, 255, 0), 'auto_rickshaw': (255, 0, 255),
            'truck': (0, 255, 255), 'van': (128, 0, 128), 'commercial_vehicle': (255, 165, 0),
            'pedestrian': (128, 128, 128), 'traffic_light': (0, 128, 255),
            'traffic_sign': (255, 128, 0), 'ambulance': (0, 0, 255)  # Red for ambulance
        }
        
        # Traffic analysis
        self.vehicle_counts = defaultdict(int)
        self.ambulance_detected = False
        self.ambulance_history = deque(maxlen=30)  # 30 frame history
        self.density_zones = self.setup_density_zones()
        
        # Performance metrics
        self.fps_counter = deque(maxlen=30)
        self.detection_stats = defaultdict(int)
        
        print("🚀 Traffic Management System initialized!")
    
    def load_config(self, config_path):
        """Load system configuration."""
        default_config = {
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "ambulance_confidence": 0.3,  # Lower threshold for ambulance
            "max_detections": 300,
            "input_size": 640,
            "enable_tracking": True,
            "density_analysis": True,
            "signal_control": False,
            "save_detections": True,
            "alert_threshold": 10  # vehicles per zone
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
        except FileNotFoundError:
            config = default_config
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config
    
    def setup_density_zones(self):
        """Setup traffic density analysis zones."""
        # Define zones for intersection analysis
        zones = {
            'north': {'x1': 0.3, 'y1': 0.0, 'x2': 0.7, 'y2': 0.3},
            'south': {'x1': 0.3, 'y1': 0.7, 'x2': 0.7, 'y2': 1.0},
            'east': {'x1': 0.7, 'y1': 0.3, 'x2': 1.0, 'y2': 0.7},
            'west': {'x1': 0.0, 'y1': 0.3, 'x2': 0.3, 'y2': 0.7},
            'center': {'x1': 0.3, 'y1': 0.3, 'x2': 0.7, 'y2': 0.7}
        }
        return zones
    
    def detect_vehicles(self, frame):
        """Detect vehicles in the current frame."""
        start_time = time.time()
        
        # Run inference
        results = self.model(
            frame,
            conf=self.config['confidence_threshold'],
            iou=self.config['iou_threshold'],
            max_det=self.config['max_detections'],
            verbose=False
        )
        
        detections = []
        ambulance_detected = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box information
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Check if class is in our mapping
                    if class_id in self.class_names:
                        class_name = self.class_names[class_id]
                        
                        # Special handling for ambulance
                        if class_id == 26:  # Ambulance
                            if confidence >= self.config['ambulance_confidence']:
                                ambulance_detected = True
                                self.detection_stats['ambulance'] += 1
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
                        self.detection_stats[class_name] += 1
        
        # Update ambulance status
        self.ambulance_history.append(ambulance_detected)
        self.ambulance_detected = sum(self.ambulance_history) > len(self.ambulance_history) * 0.3
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        self.fps_counter.append(fps)
        
        return detections, ambulance_detected
    
    def analyze_traffic_density(self, detections, frame_shape):
        """Analyze traffic density in different zones."""
        h, w = frame_shape[:2]
        zone_counts = defaultdict(int)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) / (2 * w)
            center_y = (y1 + y2) / (2 * h)
            
            # Check which zone the vehicle is in
            for zone_name, zone_coords in self.density_zones.items():
                if (zone_coords['x1'] <= center_x <= zone_coords['x2'] and
                    zone_coords['y1'] <= center_y <= zone_coords['y2']):
                    zone_counts[zone_name] += 1
        
        return zone_counts
    
    def draw_detections(self, frame, detections, zone_counts=None):
        """Draw detection boxes and information on frame."""
        h, w = frame.shape[:2]
        
        # Draw density zones (optional)
        if zone_counts and self.config.get('show_zones', True):
            for zone_name, zone_coords in self.density_zones.items():
                x1 = int(zone_coords['x1'] * w)
                y1 = int(zone_coords['y1'] * h)
                x2 = int(zone_coords['x2'] * w)
                y2 = int(zone_coords['y2'] * h)
                
                count = zone_counts.get(zone_name, 0)
                color = (0, 255, 0) if count < self.config['alert_threshold'] else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, f"{zone_name}: {count}", 
                           (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Special highlighting for ambulance
            if class_name == 'ambulance':
                thickness = 4
                color = (0, 0, 255)  # Bright red
            else:
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1-label_size[1]-5), 
                         (x1+label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_status_panel(self, frame):
        """Draw system status panel."""
        h, w = frame.shape[:2]
        
        # Create status panel
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # System status
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
        status_color = (0, 255, 0) if avg_fps > 15 else (0, 165, 255)
        
        cv2.putText(panel, f"FPS: {avg_fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Ambulance status
        ambulance_color = (0, 0, 255) if self.ambulance_detected else (128, 128, 128)
        ambulance_text = "AMBULANCE DETECTED!" if self.ambulance_detected else "No Ambulance"
        cv2.putText(panel, ambulance_text, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ambulance_color, 2)
        
        # Vehicle counts
        total_vehicles = sum(self.vehicle_counts.values())
        cv2.putText(panel, f"Total Vehicles: {total_vehicles}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Detection stats (right side)
        x_offset = w // 2
        y_offset = 25
        for i, (class_name, count) in enumerate(list(self.detection_stats.items())[:4]):
            cv2.putText(panel, f"{class_name}: {count}", 
                       (x_offset, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine with main frame
        combined = np.vstack([frame, panel])
        return combined
    
    def handle_ambulance_priority(self, zone_counts):
        """Handle ambulance priority signal control."""
        if self.ambulance_detected and self.config.get('signal_control', False):
            print("🚨 AMBULANCE DETECTED - Activating Priority Signal Control!")
            
            # Find zone with ambulance
            max_zone = max(zone_counts.items(), key=lambda x: x[1])[0]
            
            # Send signal control command (placeholder)
            signal_command = {
                'timestamp': datetime.now().isoformat(),
                'action': 'ambulance_priority',
                'zone': max_zone,
                'duration': 60  # seconds
            }
            
            # Save command (in real system, this would interface with traffic signals)
            with open('signal_commands.json', 'a') as f:
                json.dump(signal_command, f)
                f.write('\n')
            
            return signal_command
        return None
    
    def process_video(self, source=0, output_path=None):
        """Process video stream (camera or file)."""
        print(f"🎥 Starting video processing from source: {source}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video properties: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 120))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect vehicles
                detections, ambulance_in_frame = self.detect_vehicles(frame)
                
                # Analyze traffic density
                zone_counts = self.analyze_traffic_density(detections, frame.shape)
                
                # Handle ambulance priority
                if ambulance_in_frame:
                    self.handle_ambulance_priority(zone_counts)
                
                # Draw visualizations
                frame = self.draw_detections(frame, detections, zone_counts)
                frame = self.draw_status_panel(frame)
                
                # Save frame if writer is initialized
                if writer:
                    writer.write(frame)
                
                # Display frame
                cv2.imshow('Traffic Management System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    # Reset stats
                    self.detection_stats.clear()
                    print("📊 Detection stats reset")
                
                # Print periodic status
                if frame_count % 300 == 0:  # Every 10 seconds at 30fps
                    elapsed = time.time() - start_time
                    print(f"📊 Processed {frame_count} frames in {elapsed:.1f}s")
                    print(f"🚑 Ambulances detected: {self.detection_stats.get('ambulance', 0)}")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopping video processing...")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            
            print("\n📊 PROCESSING COMPLETE")
            print("=" * 50)
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {elapsed:.1f} seconds")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Ambulances detected: {self.detection_stats.get('ambulance', 0)}")
            print("\nDetection Summary:")
            for class_name, count in self.detection_stats.items():
                print(f"  {class_name}: {count}")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Traffic Management System')
    parser.add_argument('--model', '-m', required=True, 
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--source', '-s', default=0, 
                       help='Video source (camera index, video file, or RTSP stream)')
    parser.add_argument('--output', '-o', 
                       help='Output video file path (optional)')
    parser.add_argument('--config', '-c', default='config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    print("🚀 Traffic Management and Ambulance Detection System")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot") 
    print("  'r' - Reset detection stats")
    print("=" * 60)
    
    try:
        # Initialize system
        system = TrafficManagementSystem(args.model, args.config)
        
        # Process video
        system.process_video(args.source, args.output)
        
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
