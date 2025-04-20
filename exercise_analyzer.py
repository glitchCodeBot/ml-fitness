import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import numpy as np

class ExerciseAnalyzer:
    def __init__(self):
        try:
            # Load trained classifier
            self.classifier = load('models/exercise_classifier.joblib')
            
            # Initialize scaler with training data stats
            df = pd.read_csv('data/exercise_angles.csv')
            self.feature_order = [
                'Shoulder_Angle', 'Elbow_Angle', 'Hip_Angle', 
                'Knee_Angle', 'Ankle_Angle',
                'Shoulder_Ground_Angle', 'Elbow_Ground_Angle',
                'Hip_Ground_Angle', 'Knee_Ground_Angle',
                'Ankle_Ground_Angle'
            ]
            
            # Handle missing columns in training data
            for col in self.feature_order:
                if col not in df.columns:
                    df[col] = 0  # Fill missing with zeros
            
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.feature_order])
            
            # Get ideal angle ranges
            self.ideal_ranges = {
                ex: df[df['Label'] == ex][self.feature_order].describe()
                for ex in df['Label'].unique()
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize analyzer: {str(e)}")

    def classify_exercise(self, angles):
        """Classify exercise from body angles"""
        if not angles:
            return "No person detected"
        
        # Prepare features with default values for missing angles
        features = {}
        for col in self.feature_order:
            features[col] = angles.get(col, 0)  # Use 0 if angle missing
        
        # Create DataFrame and scale
        features_df = pd.DataFrame([features])[self.feature_order]
        features_scaled = self.scaler.transform(features_df)
        
        return self.classifier.predict(features_scaled)[0]

    def check_form(self, exercise_type, angles):
        """Provide form feedback"""
        if exercise_type not in self.ideal_ranges:
            return ["Unknown exercise"]
        
        feedback = []
        ranges = self.ideal_ranges[exercise_type]
        
        for angle in self.feature_order:
            if angle in angles and angle in ranges:
                value = angles[angle]
                mean = ranges[angle]['mean']
                std = ranges[angle]['std']
                
                if value < mean - 2*std:
                    feedback.append(f"{angle.replace('_', ' ')} too small")
                elif value > mean + 2*std:
                    feedback.append(f"{angle.replace('_', ' ')} too large")
        
        return feedback if feedback else ["Good form!"]
