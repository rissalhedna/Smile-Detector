import cv2
import numpy as np
import dlib
import os
import urllib.request

class SmileDetector:
    def __init__(self):
        # Path to the facial landmark predictor model
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        # Download the facial landmark predictor if it doesn't exist
        if not os.path.exists(self.predictor_path):
            print("Downloading facial landmark predictor model...")
            url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
            
            try:
                urllib.request.urlretrieve(url, self.predictor_path + ".bz2")
                
                # Extract the bz2 file
                import bz2
                with open(self.predictor_path, 'wb') as new_file, bz2.BZ2File(self.predictor_path + ".bz2", 'rb') as file:
                    for data in iter(lambda: file.read(100 * 1024), b''):
                        new_file.write(data)
                
                # Remove the bz2 file
                os.remove(self.predictor_path + ".bz2")
                print("Download complete!")
            except Exception as e:
                print(f"Error downloading model: {str(e)}")
                print("Please download the model manually from:")
                print(url)
                print(f"Extract it and place it at: {os.path.abspath(self.predictor_path)}")
                raise Exception("Failed to download the facial landmark predictor model")
        
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        
        # Initialize OpenCV's face detector as a backup
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize smile detector
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    def detect_faces(self, frame):
        """Detect faces in the frame using dlib's face detector"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        # If dlib doesn't detect any faces, try OpenCV's face detector as a backup
        if len(faces) == 0:
            opencv_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            dlib_faces = []
            for (x, y, w, h) in opencv_faces:
                dlib_faces.append(dlib.rectangle(x, y, x + w, y + h))
            return dlib_faces
        
        return faces
    
    def get_landmarks(self, frame, face):
        """Get facial landmarks for a detected face"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(gray, face)
        return landmarks
    
    def calculate_smile_probability(self, landmarks):
        """Calculate the probability of a smile based on facial landmarks"""
        # Extract mouth landmarks (indices 48-68)
        mouth_points = []
        for i in range(48, 68):
            point = (landmarks.part(i).x, landmarks.part(i).y)
            mouth_points.append(point)
        
        # Calculate mouth width and height
        mouth_width = abs(mouth_points[6][0] - mouth_points[0][0])
        mouth_height = abs((mouth_points[9][1] + mouth_points[11][1]) / 2 - 
                          (mouth_points[3][1] + mouth_points[5][1]) / 2)
        
        # Calculate smile ratio (width to height)
        if mouth_height > 0:
            smile_ratio = mouth_width / mouth_height
        else:
            smile_ratio = 0
        
        # Normalize to a probability between 0 and 1
        # These thresholds can be adjusted based on testing
        min_ratio = 2.0  # Neutral expression
        max_ratio = 5.0  # Big smile
        
        if smile_ratio <= min_ratio:
            probability = 0.0
        elif smile_ratio >= max_ratio:
            probability = 1.0
        else:
            probability = (smile_ratio - min_ratio) / (max_ratio - min_ratio)
        
        return probability
    
    def get_mouth_landmarks(self, landmarks):
        """Extract mouth landmarks for visualization"""
        mouth_points = []
        for i in range(48, 68):
            point = (landmarks.part(i).x, landmarks.part(i).y)
            mouth_points.append(point)
        return mouth_points
    
    def process_frame(self, frame):
        """Process a single frame to detect faces and smiles"""
        # Make a copy of the frame for drawing
        result_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        results = []
        for face in faces:
            # Get facial landmarks
            landmarks = self.get_landmarks(frame, face)
            
            # Calculate smile probability
            smile_probability = self.calculate_smile_probability(landmarks)
            
            # Get mouth landmarks for visualization
            mouth_points = self.get_mouth_landmarks(landmarks)
            
            # Draw face rectangle
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw mouth landmarks
            for point in mouth_points:
                cv2.circle(result_frame, point, 2, (0, 0, 255), -1)
            
            # Draw smile probability
            text = f"Smile: {smile_probability:.2f}"
            cv2.putText(result_frame, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            results.append({
                'face': (x, y, w, h),
                'smile_probability': smile_probability,
                'mouth_points': mouth_points
            })
        
        return result_frame, results
    
    def process_video(self, video_path, output_path=None, frame_callback=None):
        """Process a video file and detect smiles in each frame"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        all_results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            result_frame, frame_results = self.process_frame(frame)
            
            # Store results
            all_results.append({
                'frame_number': frame_number,
                'results': frame_results
            })
            
            # Write the frame to output video if needed
            if output_path:
                out.write(result_frame)
            
            # Call the callback function if provided
            if frame_callback:
                continue_processing = frame_callback(frame_number, frame_count, result_frame, frame_results)
                if continue_processing is False:
                    break
            
            frame_number += 1
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        
        return all_results 