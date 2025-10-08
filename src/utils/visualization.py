"""Visualization utilities."""

import cv2
import numpy as np


class Visualizer:
    """Visualization tools for results."""
    
    def __init__(self, config):
        self.config = config
        
    def draw_results(self, frame, results):
        """Draw results on frame."""
        vis_frame = frame.copy()
        
        # Draw detections
        if 'detections' in results:
            for det in results['detections']:
                if 'bbox' in det:
                    x, y, w, h = det['bbox']
                    cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw tracks
        if 'tracks' in results:
            for track in results['tracks']:
                if 'bbox' in track:
                    x, y, w, h = track['bbox']
                    track_id = track.get('id', 0)
                    cv2.putText(vis_frame, f'ID: {track_id}', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 0, 0), 2)
        
        return vis_frame
