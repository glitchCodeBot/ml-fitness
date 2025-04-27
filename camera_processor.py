import cv2
import mediapipe as mp
import numpy as np

class CameraProcessor:
    def __init__(self, camera_index=0):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=1
        )
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

    def get_frame_with_angles(self):
        success, frame = self.cap.read()
        if not success:
            return None, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return frame, None

        angles = self._calculate_all_angles(results.pose_landmarks)
        
        annotated_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.drawing_spec
        )
        
        return annotated_frame, angles

    def _calculate_all_angles(self, landmarks):
        def joint_angle(a, b, c):
            """Calculate angle between three joints"""
            a = np.array([landmarks.landmark[a].x, landmarks.landmark[a].y])
            b = np.array([landmarks.landmark[b].x, landmarks.landmark[b].y])
            c = np.array([landmarks.landmark[c].x, landmarks.landmark[c].y])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))

        def ground_angle(joint):
            """Calculate joint angle relative to vertical (ground)"""
            joint_vec = np.array([
                landmarks.landmark[joint].x,
                landmarks.landmark[joint].y - 1  # Vector pointing downward
            ])
            vertical = np.array([0, -1])  # Straight downward vector
            cosine_angle = np.dot(joint_vec, vertical) / (np.linalg.norm(joint_vec) * 1.0)
            return np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))

        return {
            # Joint angles (between body parts)
            'left_shoulder': joint_angle(11, 13, 15),
            'left_elbow': joint_angle(13, 15, 17),
            'left_hip': joint_angle(23, 25, 27),
            'left_knee': joint_angle(25, 27, 29),
            'left_ankle': joint_angle(27, 29, 31),
            
            # Ground reference angles
            'left_shoulder_ground': ground_angle(11),
            'left_elbow_ground': ground_angle(13),
            'left_hip_ground': ground_angle(23),
            'left_knee_ground': ground_angle(25),
            'left_ankle_ground': ground_angle(27)
        }

    def release(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'pose'):
            try:
                self.pose.close()
            except:
                pass