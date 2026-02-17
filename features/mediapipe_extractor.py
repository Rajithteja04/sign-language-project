import numpy as np
import mediapipe as mp
import cv2

class MediaPipeExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True
        )

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a flattened feature vector from holistic landmarks.
        TODO: define consistent ordering and normalization.
        """
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _ = self.holistic.process(image)
        # Placeholder feature vector
        return np.zeros(1662, dtype=np.float32)
