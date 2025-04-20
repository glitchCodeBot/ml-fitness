import cv2
import numpy as np
from camera_processor import CameraProcessor
from exercise_analyzer import ExerciseAnalyzer

def main():
    # Initialize components
    camera = CameraProcessor()
    analyzer = ExerciseAnalyzer()
    
    # Exercise tracking variables
    current_exercise = None
    rep_count = 0
    feedback = ""
    
    print("Starting real-time exercise tracker...")
    print("Press 'q' to quit")
    
    try:
        while True:
            frame, angles = camera.get_body_angles()
            
            if frame is None:
                break
            
            # Process frame if angles detected
            if angles:
                # Classify exercise
                exercise = analyzer.classify_exercise(angles)
                
                # Check if exercise changed
                if current_exercise != exercise:
                    current_exercise = exercise
                    rep_count = 0
                    print(f"Detected: {exercise}")
                
                # Check form and count reps
                form_feedback = analyzer.check_form(exercise, angles)
                feedback = "\n".join(form_feedback)
                
                # Simple rep counter (would need exercise-specific logic)
                if "Good form!" in form_feedback:
                    rep_count += 1
            
            # Display information
            cv2.putText(frame, f"Exercise: {current_exercise or 'None'}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Reps: {rep_count}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display feedback line by line
            for i, line in enumerate(feedback.split('\n')):
                cv2.putText(frame, line, (20, 90 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            cv2.imshow('Exercise Tracker', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nSession Summary:")
        print(f"Final Exercise: {current_exercise}")
        print(f"Total Reps: {rep_count}")

if __name__ == "__main__":
    main()
