import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
from camera_processor import CameraProcessor
from exercise_analyzer import ExerciseAnalyzer

class FitnessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Fitness Coach")
        self.root.geometry("1000x700")
        
        # Initialize components
        self.camera = CameraProcessor()
        self.analyzer = ExerciseAnalyzer()
        self.running = False
        self.current_exercise = "None"
        self.rep_count = 0
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Video frame
        self.video_frame = ttk.LabelFrame(self.root, text="Camera Feed")
        self.video_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()
        
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_processing)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Stats panel
        stats_frame = ttk.LabelFrame(self.root, text="Workout Stats")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(stats_frame, text="Current Exercise:").grid(row=0, column=0, sticky=tk.W)
        self.exercise_label = ttk.Label(stats_frame, text="None", font=('Arial', 12, 'bold'))
        self.exercise_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Rep Count:").grid(row=1, column=0, sticky=tk.W)
        self.rep_label = ttk.Label(stats_frame, text="0", font=('Arial', 12, 'bold'))
        self.rep_label.grid(row=1, column=1, sticky=tk.W)
        
        # Feedback panel
        feedback_frame = ttk.LabelFrame(self.root, text="Form Feedback")
        feedback_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        self.feedback_text = tk.Text(feedback_frame, height=5, wrap=tk.WORD)
        self.feedback_text.pack(fill=tk.BOTH, expand=True)
        self.feedback_text.insert(tk.END, "Press Start to begin analysis")
        self.feedback_text.config(state=tk.DISABLED)
    
    def start_processing(self):
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.insert(tk.END, "Starting analysis...")
        self.feedback_text.config(state=tk.DISABLED)
        
        # Start video processing in separate thread
        self.thread = threading.Thread(target=self.process_video, daemon=True)
        self.thread.start()
    
    def stop_processing(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.insert(tk.END, "\n\nSession stopped")
        self.feedback_text.config(state=tk.DISABLED)
    
    def process_video(self):
        while self.running:
            frame, angles = self.camera.get_body_angles()
            
            if frame is not None:
                # Convert frame for Tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update video feed
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Process angles if detected
                if angles:
                    exercise = self.analyzer.classify_exercise(angles)
                    feedback = self.analyzer.check_form(exercise, angles)
                    
                    # Update UI
                    self.current_exercise = exercise
                    self.rep_count += 1  # Simple rep counter - improve with exercise-specific logic
                    
                    self.exercise_label.config(text=exercise)
                    self.rep_label.config(text=str(self.rep_count))
                    
                    self.feedback_text.config(state=tk.NORMAL)
                    self.feedback_text.delete(1.0, tk.END)
                    self.feedback_text.insert(tk.END, "\n".join(feedback))
                    self.feedback_text.config(state=tk.DISABLED)
            
            # Control frame rate
            self.root.update()
            cv2.waitKey(30)
        
        # Release camera when stopped
        self.camera.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FitnessApp(root)
    root.mainloop()
