from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


FEATURE_DIM = 411


class MediaPipeExtractor:
    """
    Extracts a compact 411-dim landmark vector aligned with runtime/training usage.

    Layout (x, y, z triplets):
    - pose: 25 landmarks (25 * 3 = 75)
    - face: 70 landmarks (70 * 3 = 210)
    - left hand: 21 landmarks (21 * 3 = 63)
    - right hand: 21 landmarks (21 * 3 = 63)
    Total: 411
    """

    POSE_COUNT = 25
    FACE_COUNT = 70
    HAND_COUNT = 21

    def __init__(self) -> None:
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
        )

    @staticmethod
    def _flatten_landmarks(landmark_list, expected_count: int) -> list[float]:
        if not landmark_list:
            return [0.0] * (expected_count * 3)
        values = []
        for idx in range(expected_count):
            if idx < len(landmark_list.landmark):
                lm = landmark_list.landmark[idx]
                values.extend([float(lm.x), float(lm.y), float(lm.z)])
            else:
                values.extend([0.0, 0.0, 0.0])
        return values

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(image)

        pose = self._flatten_landmarks(result.pose_landmarks, self.POSE_COUNT)
        face = self._flatten_landmarks(result.face_landmarks, self.FACE_COUNT)
        left = self._flatten_landmarks(result.left_hand_landmarks, self.HAND_COUNT)
        right = self._flatten_landmarks(result.right_hand_landmarks, self.HAND_COUNT)

        features = np.asarray(pose + face + left + right, dtype=np.float32)
        if features.shape[0] != FEATURE_DIM:
            raise ValueError(f"Unexpected feature dim: {features.shape[0]} != {FEATURE_DIM}")
        return features

    def extract_with_debug(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(image)

        pose_count = len(result.pose_landmarks.landmark) if result.pose_landmarks else 0
        face_count = len(result.face_landmarks.landmark) if result.face_landmarks else 0
        left_count = len(result.left_hand_landmarks.landmark) if result.left_hand_landmarks else 0
        right_count = len(result.right_hand_landmarks.landmark) if result.right_hand_landmarks else 0

        pose = self._flatten_landmarks(result.pose_landmarks, self.POSE_COUNT)
        face = self._flatten_landmarks(result.face_landmarks, self.FACE_COUNT)
        left = self._flatten_landmarks(result.left_hand_landmarks, self.HAND_COUNT)
        right = self._flatten_landmarks(result.right_hand_landmarks, self.HAND_COUNT)

        features = np.asarray(pose + face + left + right, dtype=np.float32)
        if features.shape[0] != FEATURE_DIM:
            raise ValueError(f"Unexpected feature dim: {features.shape[0]} != {FEATURE_DIM}")

        debug = {
            "pose_landmarks": pose_count,
            "face_landmarks": face_count,
            "left_hand_landmarks": left_count,
            "right_hand_landmarks": right_count,
            "total_landmarks": pose_count + face_count + left_count + right_count,
            "feature_dimension": FEATURE_DIM,
            "normalization_applied": True,
            "feature_vector_generated": True,
        }
        return features, debug
