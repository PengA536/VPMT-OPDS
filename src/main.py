#!/usr/bin/env python3
"""Main entry point for volleyball player pose estimation and tracking."""

import argparse
import os
import cv2
import numpy as np
import torch
import time
from pathlib import Path

from config import Config
from models.yolov4_detector import YOLOv4Detector
from models.openpose_estimator import OpenPoseEstimator
from models.deepsort_tracker import DeepSORTTracker
from utils.visualization import Visualizer
from utils.metrics import MetricsCalculator


class VolleyballPoseTracker:
    """Main pipeline for volleyball player pose estimation and tracking."""
    
    def __init__(self, config_path=None):
        """Initialize the tracking system."""
        self.config = Config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_models()
        self.visualizer = Visualizer(self.config)
        self.metrics = MetricsCalculator()
        
    def initialize_models(self):
        """Initialize detection, pose estimation, and tracking models."""
        print("Initializing models...")
        
        # Initialize YOLOv4 detector
        self.detector = YOLOv4Detector(
            model_path=self.config.yolo_model_path,
            config_path=self.config.yolo_config_path,
            confidence_threshold=self.config.detection_confidence
        )
        
        # Initialize OpenPose estimator
        self.pose_estimator = OpenPoseEstimator(
            model_path=self.config.openpose_model_path,
            num_points=self.config.num_keypoints
        )
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSORTTracker(
            max_dist=self.config.max_cosine_distance,
            max_age=self.config.max_age
        )
        
        print(f"Models initialized on {self.device}")
    
    def process_video(self, video_path, output_path=None):
        """Process a video file for pose estimation and tracking."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        print(f"Processing {total_frames} frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_frame(frame)
            results.append(result)
            
            # Visualize and write output
            if writer:
                vis_frame = self.visualizer.draw_results(frame, result)
                writer.write(vis_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
        
        # Calculate metrics
        metrics = self.metrics.calculate_video_metrics(results)
        
        return {
            'results': results,
            'metrics': metrics,
            'fps': fps,
            'total_frames': frame_count
        }
    
    def process_frame(self, frame):
        """Process a single frame."""
        start_time = time.time()
        
        # Step 1: Detect players
        detections = self.detector.detect(frame)
        
        # Step 2: Estimate poses
        poses = []
        for det in detections:
            pose = self.pose_estimator.estimate(frame, det['bbox'])
            poses.append(pose)
        
        # Step 3: Track players
        tracks = self.tracker.update(detections, poses)
        
        processing_time = time.time() - start_time
        
        return {
            'detections': detections,
            'poses': poses,
            'tracks': tracks,
            'processing_time': processing_time
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Volleyball Player Pose Estimation and Tracking'
    )
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', help='Path to output video')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Initialize tracker
    tracker = VolleyballPoseTracker(args.config)
    
    # Process video
    results = tracker.process_video(args.video, args.output)
    
    # Print results
    print("\nProcessing complete!")
    print(f"Total frames: {results['total_frames']}")
    print(f"Average FPS: {1.0 / np.mean([r['processing_time'] for r in results['results']]):.2f}")
    print(f"Metrics: {results['metrics']}")


if __name__ == '__main__':
    main()
