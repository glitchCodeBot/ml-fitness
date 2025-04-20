import cv2
import mediapipe as mp
import numpy as np

class CameraProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=1
        )
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

    def get_body_angles(self):
        success, frame = self.cap.read()
        if not success:
            return None, None

        # Convert to RGB and process
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        if not results.pose_landmarks:
            return frame, None

        # Calculate all required angles
        angles = {}
        try:
            # Left side angles
            angles.update(self._get_left_side_angles(results.pose_landmarks))
            # Right side angles
            angles.update(self._get_right_side_angles(results.pose_landmarks))
            # Ground angles
            angles.update(self._get_ground_angles(results.pose_landmarks, frame.shape))
        except Exception as e:
            print(f"Angle calculation error: {e}")
            return frame, None

        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.drawing_spec
        )

        return frame, angles

    def _get_left_side_angles(self, landmarks):
        return {
            'Shoulder_Angle': self._calculate_angle(landmarks, 11, 13, 15),
            'Elbow_Angle': self._calculate_angle(landmarks, 13, 15, 17),
            'Hip_Angle': self._calculate_angle(landmarks, 23, 25, 27),
            'Knee_Angle': self._calculate_angle(landmarks, 25, 27, 29),
            'Ankle_Angle': self._calculate_angle(landmarks, 27, 29, 31)
        }

    def _get_right_side_angles(self, landmarks):
        return {
            'Shoulder_Angle_R': self._calculate_angle(landmarks, 12, 14, 16),
            'Elbow_Angle_R': self._calculate_angle(landmarks, 14, 16, 18),
            'Hip_Angle_R': self._calculate_angle(landmarks, 24, 26, 28),
            'Knee_Angle_R': self._calculate_angle(landmarks, 26, 28, 30),
            'Ankle_Angle_R': self._calculate_angle(landmarks, 28, 30, 32)
        }

    def _get_ground_angles(self, landmarks, frame_shape):
        ground_angles = {}
        height, width, _ = frame_shape
        
        # Vertical reference vector (downward)
        vertical = np.array([0, 1])
        
        # Calculate ground angles for left side
        for name, idx in [('Shoulder', 11), ('Elbow', 13), ('Hip', 23), 
                         ('Knee', 25), ('Ankle', 27)]:
            joint = np.array([landmarks.landmark[idx].x * width, 
                            landmarks.landmark[idx].y * height])
            vec = joint - np.array([width/2, height/2])
            cosine_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical))
            ground_angles[f"{name}_Ground_Angle"] = np.degrees(np.arccos(cosine_angle))
        
        return ground_angles

    def _calculate_angle(self, landmarks, a, b, c):
        """Calculate angle between three landmarks in 2D"""
        a = np.array([landmarks.landmark[a].x, landmarks.landmark[a].y])
        b = np.array([landmarks.landmark[b].x, landmarks.landmark[b].y])
        c = np.array([landmarks.landmark[c].x, landmarks.landmark[c].y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1, 1)  # Avoid numerical errors
        return np.degrees(np.arccos(cosine_angle))

    def release(self):
        self.cap.release()
        self.pose.close()
