import numpy as np
import cv2
from collections import deque


def preprocess_frame(frame):
    """
    Convert RGB frame to grayscale 84x84 normalized image
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame


class FrameStack:
    """
    Maintain a fixed-length stack of frames
    Output shape: (num_frames, 84, 84)
    """

    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        """
        Initialize stack with the same frame repeated
        """
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self.get_state()

    def append(self, frame):
        self.frames.append(frame)
        return self.get_state()

    def get_state(self):
        return np.stack(self.frames, axis=0)
