"""Configuration management."""

class Config:
    """Configuration class for system parameters."""
    
    def __init__(self, config_path=None):
        # Model paths
        self.yolo_model_path = 'models/yolov4.weights'
        self.yolo_config_path = 'models/yolov4.cfg'
        self.openpose_model_path = 'models/openpose.pth'
        
        # Parameters
        self.detection_confidence = 0.5
        self.num_keypoints = 18
        self.max_cosine_distance = 0.3
        self.max_age = 30
        
        # Training
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.epochs = 30
