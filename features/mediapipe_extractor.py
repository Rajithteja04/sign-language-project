import cv2
import mediapipe as mp
import numpy as np

from utils.keypoint_format import (
    FACE_POINTS,
    FEATURE_DIM,
    HAND_POINTS,
    POSE_POINTS,
    flatten_points,
    normalize_openpose_like,
)


class MediaPipeExtractor:
    FACE70_MAP = np.linspace(0, 467, FACE_POINTS, dtype=np.int32).tolist()

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True
        )

    @staticmethod
    def _pick(lms, idx: int):
        if lms is None or idx >= len(lms.landmark):
            return 0.0, 0.0, 0.0
        lm = lms.landmark[idx]
        return float(lm.x), float(lm.y), 1.0

    @staticmethod
    def _pick_pose(lms, idx: int):
        if lms is None or idx >= len(lms.landmark):
            return 0.0, 0.0, 0.0
        lm = lms.landmark[idx]
        return float(lm.x), float(lm.y), float(lm.visibility)

    def _build_pose25(self, results) -> np.ndarray:
        pose = np.zeros((POSE_POINTS, 3), dtype=np.float32)
        lms = results.pose_landmarks

        mapping = {
            0: ("pose", 0),
            2: ("pose", 12),
            3: ("pose", 14),
            4: ("pose", 16),
            5: ("pose", 11),
            6: ("pose", 13),
            7: ("pose", 15),
            9: ("pose", 24),
            10: ("pose", 26),
            11: ("pose", 28),
            12: ("pose", 23),
            13: ("pose", 25),
            14: ("pose", 27),
            15: ("pose", 5),
            16: ("pose", 2),
            17: ("pose", 8),
            18: ("pose", 7),
            19: ("pose", 31),
            20: ("pose", 31),
            21: ("pose", 29),
            22: ("pose", 32),
            23: ("pose", 32),
            24: ("pose", 30),
        }

        for op_idx, (_, mp_idx) in mapping.items():
            pose[op_idx] = self._pick_pose(lms, mp_idx)

        # neck: midpoint of shoulders
        rs = pose[2]
        ls = pose[5]
        if rs[2] > 0 and ls[2] > 0:
            pose[1] = np.array([(rs[0] + ls[0]) * 0.5, (rs[1] + ls[1]) * 0.5, (rs[2] + ls[2]) * 0.5], dtype=np.float32)

        # mid-hip: midpoint of hips
        rh = pose[9]
        lh = pose[12]
        if rh[2] > 0 and lh[2] > 0:
            pose[8] = np.array([(rh[0] + lh[0]) * 0.5, (rh[1] + lh[1]) * 0.5, (rh[2] + lh[2]) * 0.5], dtype=np.float32)

        return pose

    def _build_face70(self, results) -> np.ndarray:
        face = np.zeros((FACE_POINTS, 3), dtype=np.float32)
        lms = results.face_landmarks
        for i, idx in enumerate(self.FACE70_MAP):
            face[i] = self._pick(lms, idx)
        return face

    @staticmethod
    def _build_hand21(hand_landmarks) -> np.ndarray:
        hand = np.zeros((HAND_POINTS, 3), dtype=np.float32)
        if hand_landmarks is None:
            return hand
        for i in range(min(HAND_POINTS, len(hand_landmarks.landmark))):
            lm = hand_landmarks.landmark[i]
            hand[i] = np.array([float(lm.x), float(lm.y), 1.0], dtype=np.float32)
        return hand

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extract normalized OpenPose-like feature vector:
        pose25 + face70 + left hand21 + right hand21, each as (x, y, conf).
        """
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True

        pose = self._build_pose25(results)
        face = self._build_face70(results)
        left = self._build_hand21(results.left_hand_landmarks)
        right = self._build_hand21(results.right_hand_landmarks)
        points = np.vstack([pose, face, left, right]).astype(np.float32)
        points = normalize_openpose_like(points)
        vec = flatten_points(points, use_confidence=True)

        if vec.shape[0] != FEATURE_DIM:
            raise RuntimeError(f"Expected runtime feature dim {FEATURE_DIM}, got {vec.shape[0]}")
        return vec
