"""
Real-Time Traffic Management System - Full Deployment Version
============================================================

This script provides a complete real-time traffic management system with
advanced ambulance detection, traffic density analysis, and signal control.

Features:
- Real-time video processing (camera/file)
- Advanced ambulance detection with priority handling
- Traffic density analysis and flow optimization
- Signal timing optimization
- Multi-lane traffic monitoring
- Performance analytics and logging
- Web dashboard integration ready

Author: Traffic Management System Team
Date: September 2025
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import json
import logging
from datetime import datetime
from collections import deque, defaultdict
import threading
import queue
import argparse
from pathlib import Path
import sqlite3
import pandas as pd

class TrafficDatabase:
    """Database handler for traffic analytics."""
    
    def __init__(self, db_path="traffic_analytics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Traffic events table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            vehicle_count INTEGER,
            ambulance_count INTEGER,
            avg_confidence REAL,
            lane_id TEXT,
            signal_state TEXT
        )
        ''')
        
        # Ambulance alerts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ambulance_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            position_x REAL,
            position_y REAL,
            lane_id TEXT,
            response_time REAL,
            priority_given BOOLEAN
        )
        ''')
        
        # Performance metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            fps REAL,
            processing_time REAL,
            detection_count INTEGER,
            memory_usage REAL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_traffic_event(self, event_type, vehicle_count, ambulance_count, 
                         avg_confidence, lane_id="main", signal_state="normal"):
        """Log traffic event to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO traffic_events 
        (event_type, vehicle_count, ambulance_count, avg_confidence, lane_id, signal_state)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (event_type, vehicle_count, ambulance_count, avg_confidence, lane_id, signal_state))
        
        conn.commit()
        conn.close()
    
    def log_ambulance_alert(self, confidence, pos_x, pos_y, lane_id, response_time, priority_given):
        """Log ambulance detection alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO ambulance_alerts 
        (confidence, position_x, position_y, lane_id, response_time, priority_given)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (confidence, pos_x, pos_y, lane_id, response_time, priority_given))
        
        conn.commit()
        conn.close()

class SignalController:
    """Traffic signal control system."""
    
    def __init__(self):
        self.current_state = "normal"
        self.normal_timing = {"green": 30, "yellow": 5, "red": 30}
        self.emergency_timing = {"green": 60, "yellow": 3, "red": 15}
        self.signal_start_time = time.time()
        self.current_phase = "green"
        self.emergency_mode = False
        self.emergency_start_time = None
        
    def update_signal(self, ambulance_detected=False):
        """Update traffic signal based on conditions."""
        current_time = time.time()
        
        if ambulance_detected and not self.emergency_mode:
            # Switch to emergency mode
            self.emergency_mode = True
            self.emergency_start_time = current_time
            self.current_phase = "green"
            self.signal_start_time = current_time
            self.current_state = "emergency"
            return True  # Signal changed
        
        # Check if emergency mode should end
        if self.emergency_mode and (current_time - self.emergency_start_time > 120):  # 2 minutes
            self.emergency_mode = False
            self.current_state = "normal"
            self.signal_start_time = current_time
        
        # Normal signal timing logic
        timing = self.emergency_timing if self.emergency_mode else self.normal_timing
        phase_duration = current_time - self.signal_start_time
        
        if self.current_phase == "green" and phase_duration > timing["green"]:
            self.current_phase = "yellow"
            self.signal_start_time = current_time
        elif self.current_phase == "yellow" and phase_duration > timing["yellow"]:
            self.current_phase = "red"
            self.signal_start_time = current_time
        elif self.current_phase == "red" and phase_duration > timing["red"]:
            self.current_phase = "green"
            self.signal_start_time = current_time
        
        return False  # No emergency change

class AdvancedTrafficSystem:
    """Advanced real-time traffic management system."""
    
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize the traffic management system."""
        print("🚦 Initializing Advanced Traffic Management System...")
        
        # Load YOLO model
        print(f"🔄 Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        self.confidence_threshold = confidence_threshold
        
        # Class mapping
        self.class_names = {
            0: 'car', 2: 'motorcycle', 3: 'bus', 5: 'bicycle',
            6: 'auto_rickshaw', 7: 'truck', 8: 'van', 9: 'commercial_vehicle',
            10: 'pedestrian', 22: 'traffic_light', 24: 'traffic_sign',
            26: 'ambulance'
        }
        
        # Colors for visualization
        self.colors = {
            'car': (255, 0, 0), 'motorcycle': (0, 255, 0), 'bus': (0, 0, 255),
            'bicycle': (255, 255, 0), 'auto_rickshaw': (255, 0, 255),
            'truck': (0, 255, 255), 'van': (128, 0, 128), 'commercial_vehicle': (255, 165, 0),
            'pedestrian': (0, 128, 255), 'traffic_light': (128, 128, 128),
            'traffic_sign': (64, 64, 64), 'ambulance': (0, 0, 255)
        }
        
        # Traffic analysis
        self.frame_buffer = deque(maxlen=30)  # Store 1 second of frames (30 FPS)
        self.vehicle_counts = defaultdict(int)
        self.traffic_density = 0
        self.ambulance_alerts = deque(maxlen=10)
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.processing_times = deque(maxlen=100)
        
        # Signal controller
        self.signal_controller = SignalController()
        
        # Database
        self.db = TrafficDatabase()
        
        # Setup logging
        self.setup_logging()
        
        # Analytics
        self.total_vehicles_detected = 0
        self.total_ambulances_detected = 0
        self.session_start_time = time.time()
        
        print("✅ Traffic Management System initialized!")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('traffic_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame using YOLO."""
        start_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        detections = []
        ambulance_detected = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box information
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if cls in self.class_names:
                        class_name = self.class_names[cls]
                        
                        detection = {
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': xyxy,
                            'center': ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                        }
                        detections.append(detection)
                        
                        # Track vehicle counts
                        self.vehicle_counts[class_name] += 1
                        self.total_vehicles_detected += 1
                        
                        # Special handling for ambulance
                        if class_name == 'ambulance':
                            ambulance_detected = True
                            self.total_ambulances_detected += 1
                            
                            # Log ambulance alert
                            alert = {
                                'timestamp': time.time(),
                                'confidence': conf,
                                'position': detection['center'],
                                'bbox': xyxy
                            }
                            self.ambulance_alerts.append(alert)
                            
                            # Database logging
                            self.db.log_ambulance_alert(
                                conf, detection['center'][0], detection['center'][1],
                                "main", processing_time, True
                            )
                            
                            self.logger.warning(f"🚑 AMBULANCE DETECTED! Confidence: {conf:.3f}")
        
        return detections, ambulance_detected, processing_time
    
    def analyze_traffic_density(self, detections):
        """Analyze traffic density and flow."""
        vehicle_detections = [d for d in detections if d['class_name'] != 'pedestrian']
        current_density = len(vehicle_detections)
        
        # Update density buffer
        self.frame_buffer.append(current_density)
        
        # Calculate moving average
        if self.frame_buffer:
            self.traffic_density = sum(self.frame_buffer) / len(self.frame_buffer)
        
        # Classify traffic level
        if self.traffic_density < 3:
            traffic_level = "Light"
        elif self.traffic_density < 8:
            traffic_level = "Moderate"
        elif self.traffic_density < 15:
            traffic_level = "Heavy"
        else:
            traffic_level = "Congested"
        
        return {
            'current_count': current_density,
            'avg_density': self.traffic_density,
            'level': traffic_level,
            'vehicle_types': {d['class_name']: sum(1 for det in vehicle_detections 
                                                 if det['class_name'] == d['class_name']) 
                           for d in vehicle_detections}
        }
    
    def draw_detections(self, frame, detections, traffic_analysis, ambulance_detected):
        """Draw detection boxes and information on frame."""
        height, width = frame.shape[:2]
        
        # Draw detection boxes
        for detection in detections:
            bbox = detection['bbox'].astype(int)
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Special handling for ambulance
            if class_name == 'ambulance':
                color = (0, 0, 255)  # Red
                thickness = 4
                # Draw flashing effect
                if int(time.time() * 4) % 2:  # Flash every 0.25 seconds
                    cv2.rectangle(frame, (bbox[0]-5, bbox[1]-5), 
                                (bbox[2]+5, bbox[3]+5), (0, 0, 255), 6)
            else:
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw info panel
        self.draw_info_panel(frame, traffic_analysis, ambulance_detected)
        
        return frame
    
    def draw_info_panel(self, frame, traffic_analysis, ambulance_detected):
        """Draw information panel on frame."""
        height, width = frame.shape[:2]
        panel_height = 200
        panel_width = 400
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # System title
        cv2.putText(frame, "TRAFFIC MANAGEMENT SYSTEM", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Traffic density
        density_color = (0, 255, 0) if traffic_analysis['level'] == "Light" else \
                       (0, 255, 255) if traffic_analysis['level'] == "Moderate" else \
                       (0, 165, 255) if traffic_analysis['level'] == "Heavy" else (0, 0, 255)
        
        cv2.putText(frame, f"Traffic: {traffic_analysis['level']}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, density_color, 2)
        cv2.putText(frame, f"Vehicles: {traffic_analysis['current_count']}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Ambulance status
        if ambulance_detected:
            cv2.putText(frame, "🚨 AMBULANCE PRIORITY 🚨", (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Signal status
        signal_color = (0, 255, 0) if self.signal_controller.current_phase == "green" else \
                      (0, 255, 255) if self.signal_controller.current_phase == "yellow" else (0, 0, 255)
        
        signal_text = f"Signal: {self.signal_controller.current_phase.upper()}"
        if self.signal_controller.emergency_mode:
            signal_text += " (EMERGENCY)"
        
        cv2.putText(frame, signal_text, (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, signal_color, 2)
        
        # Performance metrics
        if self.fps_buffer:
            avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 185),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Session statistics (right panel)
        stats_x = width - 300
        cv2.rectangle(frame, (stats_x, 10), (width - 10, 150), (0, 0, 0), -1)
        
        session_time = time.time() - self.session_start_time
        hours = int(session_time // 3600)
        minutes = int((session_time % 3600) // 60)
        
        cv2.putText(frame, "SESSION STATS", (stats_x + 10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Runtime: {hours:02d}:{minutes:02d}", (stats_x + 10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Total Vehicles: {self.total_vehicles_detected}", (stats_x + 10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Ambulances: {self.total_ambulances_detected}", (stats_x + 10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Alerts: {len(self.ambulance_alerts)}", (stats_x + 10, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        """Process single frame through the system."""
        # Detect vehicles
        detections, ambulance_detected, processing_time = self.detect_vehicles(frame)
        
        # Analyze traffic
        traffic_analysis = self.analyze_traffic_density(detections)
        
        # Update signal controller
        signal_changed = self.signal_controller.update_signal(ambulance_detected)
        
        # Log to database periodically
        if int(time.time()) % 10 == 0:  # Every 10 seconds
            avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0
            self.db.log_traffic_event(
                "detection", len(detections), 
                sum(1 for d in detections if d['class_name'] == 'ambulance'),
                avg_conf, "main", self.signal_controller.current_state
            )
        
        # Draw visualization
        output_frame = self.draw_detections(frame, detections, traffic_analysis, ambulance_detected)
        
        # Update FPS
        fps = 1.0 / processing_time if processing_time > 0 else 0
        self.fps_buffer.append(fps)
        
        return output_frame, {
            'detections': detections,
            'traffic_analysis': traffic_analysis,
            'ambulance_detected': ambulance_detected,
            'signal_changed': signal_changed,
            'fps': fps,
            'processing_time': processing_time
        }
    
    def run_camera(self, camera_id=0):
        """Run system with camera input."""
        print(f"📹 Starting camera feed (Camera ID: {camera_id})...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Could not access camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized. Press 'q' to quit.")
        print("Keys: 'q'=quit, 's'=save screenshot, 'r'=reset stats")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from camera")
                    break
                
                # Process frame
                output_frame, stats = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Traffic Management System', output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"traffic_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, output_frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):
                    # Reset statistics
                    self.total_vehicles_detected = 0
                    self.total_ambulances_detected = 0
                    self.session_start_time = time.time()
                    print("🔄 Statistics reset")
        
        except KeyboardInterrupt:
            print("\n⏹️  System stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera released and windows closed")
    
    def run_video(self, video_path):
        """Run system with video file input."""
        print(f"🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        print(f"📊 Video info: {frame_count} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        print("✅ Processing started. Press 'q' to quit, 'p' to pause.")
        
        frame_num = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("✅ Video processing completed!")
                        break
                    
                    frame_num += 1
                    
                    # Process frame
                    output_frame, stats = self.process_frame(frame)
                    
                    # Add progress bar
                    progress = frame_num / frame_count
                    bar_width = 400
                    bar_height = 20
                    bar_x = 50
                    bar_y = output_frame.shape[0] - 40
                    
                    # Background
                    cv2.rectangle(output_frame, (bar_x, bar_y), 
                                (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                    # Progress
                    progress_width = int(bar_width * progress)
                    cv2.rectangle(output_frame, (bar_x, bar_y), 
                                (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
                    # Text
                    cv2.putText(output_frame, f"{progress:.1%} ({frame_num}/{frame_count})",
                               (bar_x + 10, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Traffic Management System - Video Processing', output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"traffic_video_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, output_frame)
                    print(f"📸 Frame saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Processing stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Video processing finished")
    
    def generate_analytics_report(self):
        """Generate analytics report from database."""
        print("📊 Generating analytics report...")
        
        conn = sqlite3.connect(self.db.db_path)
        
        # Traffic events summary
        events_df = pd.read_sql_query("SELECT * FROM traffic_events", conn)
        ambulance_df = pd.read_sql_query("SELECT * FROM ambulance_alerts", conn)
        
        report = {
            'session_summary': {
                'total_runtime': time.time() - self.session_start_time,
                'total_vehicles_detected': self.total_vehicles_detected,
                'total_ambulances_detected': self.total_ambulances_detected,
                'avg_fps': sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0
            },
            'database_summary': {
                'total_traffic_events': len(events_df),
                'total_ambulance_alerts': len(ambulance_df),
                'avg_vehicles_per_event': events_df['vehicle_count'].mean() if not events_df.empty else 0
            }
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"traffic_analytics_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Analytics report saved: {report_file}")
        
        conn.close()
        return report

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Advanced Traffic Management System')
    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--source', '-s', default='0',
                       help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--output', '-o',
                       help='Output video file path (optional)')
    
    args = parser.parse_args()
    
    print("🚦 Advanced Traffic Management System")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.confidence}")
    print("=" * 60)
    
    try:
        # Initialize system
        system = AdvancedTrafficSystem(args.model, args.confidence)
        
        # Determine source type and run
        if args.source.isdigit():
            # Camera input
            camera_id = int(args.source)
            system.run_camera(camera_id)
        else:
            # Video file input
            system.run_video(args.source)
        
        # Generate final report
        print("\n📊 Generating final analytics report...")
        report = system.generate_analytics_report()
        
        print("\n🎉 SYSTEM SHUTDOWN COMPLETE!")
        print("=" * 60)
        print(f"📊 Total vehicles detected: {system.total_vehicles_detected}")
        print(f"🚑 Total ambulances detected: {system.total_ambulances_detected}")
        if system.fps_buffer:
            avg_fps = sum(system.fps_buffer) / len(system.fps_buffer)
            print(f"⚡ Average FPS: {avg_fps:.1f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ System error: {e}")
        logging.exception("System error occurred")
        return 1

if __name__ == "__main__":
    exit(main())
