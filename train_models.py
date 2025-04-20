import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump

def train_exercise_classifier():
    # Load exercise angle data
    df = pd.read_csv('data/exercise_angles.csv')
    
    # Feature engineering
    features = df.drop(['Side', 'Label'], axis=1)
    labels = df['Label']
    
    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print(f"Exercise Classifier Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Save model
    dump(clf, 'models/exercise_classifier.joblib')
    print("Exercise classifier trained and saved successfully!")

def train_calorie_predictor():
    # Load user workout data
    df = pd.read_csv('data/user_workouts.csv')
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Separate features and target
    features = df.drop(['User ID', 'Calories Burned'], axis=1)
    target = df['Calories Burned']
    
    # Train regressor
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    
    # Evaluate
    y_pred = reg.predict(X_test)
    print(f"Calorie Predictor MAE: {mean_absolute_error(y_test, y_pred):.2f} calories")
    
    # Save model and encoders
    dump({'model': reg, 'encoders': label_encoders}, 'models/calorie_predictor.joblib')
    print("Calorie predictor trained and saved successfully!")

if __name__ == "__main__":
    print("Training exercise classifier...")
    train_exercise_classifier()
    
    print("\nTraining calorie predictor...")
    train_calorie_predictor()