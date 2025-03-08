# Smile Detector

This application detects smiles in human faces from uploaded videos. It uses pre-trained models to:

1. Detect faces in each video frame
2. Identify facial landmarks
3. Calculate the probability of a smile
4. Visualize the facial landmarks around the mouth

## Setup

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Download the required pre-trained models:

   - The face detector model (included in OpenCV)
   - The facial landmark predictor from dlib (will be downloaded automatically on first run)

3. Run the application:

   ```
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:5000`

5. Upload a video file and the application will process it, showing the smile detection results.

## How It Works

The application uses:

- OpenCV's Haar Cascade classifier for face detection
- dlib's facial landmark predictor to identify 68 facial landmarks
- Custom smile detection logic based on the geometry of mouth landmarks
- Flask for the web interface

## Requirements

- Python 3.7+
- See requirements.txt for all dependencies
