import cv2
import time
from camera_processor import CameraProcessor
from exercise_analyzer import ExerciseAnalyzer

def display_info(frame, exercise, rep_count, feedback):
    """Helper function to display information on frame"""
    y_offset = 40
    cv2.putText(frame, f"Exercise: {exercise}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"Reps: {rep_count}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for msg in feedback:
        y_offset += 30
        color = (0, 255, 0) if "Good" in msg else (0, 0, 255)
        cv2.putText(frame, msg, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

def main():
    camera = CameraProcessor()
    analyzer = ExerciseAnalyzer()
    last_exercise = None
    
    print("Starting Virtual Fitness Coach. Press 'q' to quit...")
    start_time = time.time()
    
    try:
        while True:
            frame, angles = camera.get_frame_with_angles()
            
            if frame is None:
                continue
                
            analysis = analyzer.analyze_movement(angles)
            
            # Reset counter if exercise changed
            current_exercise = analysis["exercise"]
            if current_exercise != last_exercise:
                analyzer = ExerciseAnalyzer()
                last_exercise = current_exercise
            
            display_info(frame, current_exercise, analysis["count"], analysis["feedback"])
            cv2.imshow('Virtual Fitness Coach', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.release()
        cv2.destroyAllWindows()
        duration = time.time() - start_time
        print(f"\nWorkout Complete! Duration: {duration//60:.0f}m {duration%60:.0f}s")
        print(f"Total reps: {analyzer.rep_count}")

if __name__ == "__main__":
    main()