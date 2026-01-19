import cv2
import numpy as np
from collections import deque

def preprocess_frame(frame):
    """Convert 210x160x3 RGB to 84x84x1 grayscale"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

class FrameStack:
    """Stack last N frames"""
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        processed = preprocess_frame(frame)
        for _ in range(self.num_frames):
            self.frames.append(processed)
        return self._get_state()
    
    def step(self, frame):
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return self._get_state()
    
    def _get_state(self):
        return np.stack(self.frames, axis=0)
