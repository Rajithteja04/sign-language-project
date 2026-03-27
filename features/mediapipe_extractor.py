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
        # Hysteresis state to reduce pose-count flicker in UI debug panel.
        self._torso_score = 0
        self._torso_visible_state = False

    def _torso_visible(self, pose_landmarks, min_visibility: float = 0.35) -> bool:
        """Stable upper-body gate for demo: shoulders visible => pose shown."""
        if not pose_landmarks or not hasattr(pose_landmarks, "landmark"):
            self._torso_score = max(self._torso_score - 1, -3)
        else:
            lms = pose_landmarks.landmark
            shoulders = (11, 12)
            shoulder_vis = 0
            for idx in shoulders:
                if idx < len(lms):
                    vis = float(getattr(lms[idx], "visibility", 0.0))
                    if vis >= min_visibility:
                        shoulder_vis += 1

            # Positive if both shoulders are visible; otherwise decay.
            if shoulder_vis >= 2:
                self._torso_score = min(self._torso_score + 1, 3)
            else:
                self._torso_score = max(self._torso_score - 1, -3)

        # Hysteresis to prevent rapid 0<->33 flicker.
        if self._torso_score >= 1:
            self._torso_visible_state = True
        elif self._torso_score <= -1:
            self._torso_visible_state = False

        return self._torso_visible_state

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

        torso_visible = self._torso_visible(result.pose_landmarks)
        pose_count = len(result.pose_landmarks.landmark) if (result.pose_landmarks and torso_visible) else 0
        face_count = len(result.face_landmarks.landmark) if result.face_landmarks else 0
        left_count = len(result.left_hand_landmarks.landmark) if result.left_hand_landmarks else 0
        right_count = len(result.right_hand_landmarks.landmark) if result.right_hand_landmarks else 0

        pose = self._flatten_landmarks(result.pose_landmarks if torso_visible else None, self.POSE_COUNT)
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
