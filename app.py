import os
import uuid
import threading
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from smile_detector import SmileDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Initialize the smile detector
detector = SmileDetector()

# Store processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video_task(video_id, input_path, output_path):
    """Background task to process a video"""
    try:
        processing_status[video_id]['status'] = 'processing'
        
        # Define a callback to update progress
        def progress_callback(frame_number, total_frames, frame, results):
            progress = int((frame_number / total_frames) * 100)
            processing_status[video_id]['progress'] = progress
        
        # Process the video
        detector.process_video(input_path, output_path, progress_callback)
        
        processing_status[video_id]['status'] = 'completed'
        processing_status[video_id]['progress'] = 100
        processing_status[video_id]['output_path'] = output_path
    except Exception as e:
        processing_status[video_id]['status'] = 'error'
        processing_status[video_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for this upload
        video_id = str(uuid.uuid4())
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        base_filename, extension = os.path.splitext(filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}{extension}")
        file.save(input_path)
        
        # Define output path
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{video_id}_processed.mp4")
        
        # Initialize processing status
        processing_status[video_id] = {
            'status': 'queued',
            'progress': 0,
            'filename': filename,
            'input_path': input_path,
            'output_path': None
        }
        
        # Start processing in a background thread
        thread = threading.Thread(
            target=process_video_task,
            args=(video_id, input_path, output_path)
        )
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('processing', video_id=video_id))
    
    return redirect(request.url)

@app.route('/processing/<video_id>')
def processing(video_id):
    if video_id not in processing_status:
        return redirect(url_for('index'))
    
    return render_template('processing.html', video_id=video_id, 
                          filename=processing_status[video_id]['filename'])

@app.route('/status/<video_id>')
def status(video_id):
    if video_id not in processing_status:
        return jsonify({'status': 'not_found'})
    
    return jsonify(processing_status[video_id])

@app.route('/result/<video_id>')
def result(video_id):
    if video_id not in processing_status or processing_status[video_id]['status'] != 'completed':
        return redirect(url_for('index'))
    
    return render_template('result.html', video_id=video_id, 
                          filename=processing_status[video_id]['filename'])

@app.route('/video/<video_id>')
def video(video_id):
    if video_id not in processing_status or 'output_path' not in processing_status[video_id]:
        return redirect(url_for('index'))
    
    return send_from_directory(app.config['PROCESSED_FOLDER'], 
                              os.path.basename(processing_status[video_id]['output_path']))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 