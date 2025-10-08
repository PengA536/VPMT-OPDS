"""DeepSORT tracker implementation."""

import numpy as np
from scipy.optimize import linear_sum_assignment


class DeepSORTTracker:
    """DeepSORT tracker for multi-object tracking."""
    
    def __init__(self, max_dist=0.3, max_age=30):
        self.max_dist = max_dist
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1
        
    def update(self, detections, poses):
        """Update tracks with new detections."""
        active_tracks = []
        for i, det in enumerate(detections):
            active_tracks.append({
                'id': self.next_id,
                'bbox': det.get('bbox', [0, 0, 100, 200]),
                'pose': poses[i] if i < len(poses) else None,
                'confidence': det.get('confidence', 0.9)
            })
            self.next_id += 1
        return active_tracks
