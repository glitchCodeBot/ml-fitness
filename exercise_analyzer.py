import numpy as np
import pandas as pd
from collections import deque
import time
from joblib import load
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class ExerciseAnalyzer:
    def __init__(self):
        model_path = os.path.join('models', 'exercise_classifier.joblib')
        if os.path.exists(model_path):
            self.classifier = load(model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.rep_count = 0
        self.stage = "rest"
        self.angle_history = deque(maxlen=5)
        self.last_rep_time = 0
        
        self.exercise_config = {
            "Jumping Jacks": {
                "primary_joint": "left_shoulder",
                "min_angle": 45,
                "max_angle": 150,
                "cooldown": 0.8,
                "form_rules": {
                    "arms_extended": lambda a: a["left_elbow"] > 160,
                    "legs_spread": lambda a: a["left_hip_ground"] < 70
                }
            },
            # Add other exercises as needed
        }

    def classify_exercise(self, angles):
        if not angles:
            return "unknown"
        
        try:
            features = np.array([[
                angles['left_shoulder'],      # Shoulder_Angle
                angles['left_elbow'],         # Elbow_Angle
                angles['left_hip'],           # Hip_Angle
                angles['left_knee'],          # Knee_Angle
                angles['left_ankle'],         # Ankle_Angle
                angles['left_shoulder_ground'],  # Shoulder_Ground_Angle
                angles['left_elbow_ground'],    # Elbow_Ground_Angle
                angles['left_hip_ground'],      # Hip_Ground_Angle
                angles['left_knee_ground'],     # Knee_Ground_Angle
                angles['left_ankle_ground']     # Ankle_Ground_Angle
            ]])
            
            return self.classifier.predict(features)[0]
        except Exception as e:
            print(f"Classification error: {e}")
            return "unknown"

    def analyze_movement(self, angles):
        if not angles:
            return {
                "exercise": "none",
                "feedback": ["No person detected"],
                "count": self.rep_count
            }
        
        exercise = self.classify_exercise(angles)
        feedback = []
        counted_rep = False
        
        if exercise in self.exercise_config:
            cfg = self.exercise_config[exercise]
            primary_angle = angles.get(cfg["primary_joint"], 0)
            self.angle_history.append(primary_angle)
            
            # Form checking
            for rule_name, rule_func in cfg["form_rules"].items():
                if not rule_func(angles):
                    feedback.append(f"Improve {rule_name.replace('_', ' ')}")
            
            # Rep counting
            current_time = time.time()
            if (self.stage == "rest" and primary_angle < cfg["min_angle"] and
                current_time - self.last_rep_time > cfg["cooldown"]):
                self.stage = "active"
                
            elif (self.stage == "active" and primary_angle > cfg["max_angle"] and
                  not feedback):  # Only count if form is good
                self.rep_count += 1
                counted_rep = True
                self.last_rep_time = current_time
                feedback.append("Good rep!")
                self.stage = "rest"
        
        return {
            "exercise": exercise,
            "feedback": feedback if feedback else ["Good form!"],
            "count": self.rep_count,
            "new_rep": counted_rep
        }