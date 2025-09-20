import os
import csv
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date
import base64
import io
from io import BytesIO
from PIL import Image
import logging
import time

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response, session, send_from_directory, make_response
from flask_cors import CORS
import deepface
from deepface import DeepFace

# Import configuration
try:
    from config import *
except ImportError:
    # Fallback configuration if config.py is not found
    MODEL_NAME = "Facenet"
    DETECTOR_BACKEND = "retinaface"
    DISTANCE_METRIC = "cosine"
    RECOGNITION_THRESHOLD = 0.4
    AUTO_ATTENDANCE = True
    SECRET_KEY = 'your-secret-key-here'
    DEBUG_MODE = True
    HOST = '0.0.0.0'
    PORT = 5000
    UPLOAD_FOLDER = 'static/uploads'
    EMBEDDINGS_FOLDER = 'static/embeddings'
    TEMP_FOLDER = 'static/temp'
    STUDENT_DATA_CSV = 'students.csv'
    ATTENDANCE_CSV = 'attendance.csv'

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('face_recognition_system.log')
    ]
)
logger = logging.getLogger(__name__)
logger.info("Face Recognition System starting up...")

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global variables
face_recognition_model = None
students_data = {}

class FaceRecognitionSystem:
    def __init__(self):
        # Load configuration with proper error handling
        try:
            self.model_name = MODEL_NAME
            self.detector_backend = DETECTOR_BACKEND
            self.distance_metric = DISTANCE_METRIC
            self.threshold = RECOGNITION_THRESHOLD
            logger.info(f"Face recognition configured with: Model={self.model_name}, Detector={self.detector_backend}, Threshold={self.threshold}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Fallback to safe defaults
            self.model_name = "Facenet"
            self.detector_backend = "opencv"  # More reliable fallback
            self.distance_metric = "cosine"
            self.threshold = 0.4  # Lower threshold for better matching
            logger.info(f"Using fallback configuration: Model={self.model_name}, Detector={self.detector_backend}, Threshold={self.threshold}")
            
        # Try different detector backends if the primary one fails
        self.detector_backends = [self.detector_backend, "opencv", "mtcnn", "ssd"]
        self.current_detector_index = 0
            
        self.students_data = {}
        self.load_model()
        
    def load_model(self):
        """Load the pretrained FaceNet/DeepFace model"""
        try:
            # This will automatically download and load the Facenet model
            logger.info(f"Loading DeepFace model ({self.model_name})...")
            # Build the model which will cache it for subsequent use
            model = DeepFace.build_model(self.model_name)
            
            # Check if model was actually loaded
            if model is None:
                logger.error("Failed to build DeepFace model - returned None")
                return False
                
            logger.info(f"DeepFace model loaded successfully: {self.model_name}")
            
            # Verify detector backend
            detector_backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
            if self.detector_backend not in detector_backends:
                logger.warning(f"Unknown detector backend: {self.detector_backend}. Falling back to retinaface.")
                self.detector_backend = "retinaface"
                
            logger.info(f"Using detector backend: {self.detector_backend}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_students(self):
        """Load student data from CSV file"""
        self.students_data = {}
        
        if not os.path.exists(STUDENT_DATA_CSV):
            # Create CSV file with headers if it doesn't exist
            with open(STUDENT_DATA_CSV, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Student_ID', 'Name', 'Email', 'Phone', 'Embedding_Path', 'Image_Path', 'Created_Date'])
            logger.info("Created new students.csv file")
            return {}
        
        try:
            df = pd.read_csv(STUDENT_DATA_CSV)
            for _, row in df.iterrows():
                student_id = str(row['Student_ID'])
                self.students_data[student_id] = {
                    'name': row['Name'],
                    'email': row.get('Email', ''),
                    'phone': row.get('Phone', ''),
                    'embedding_path': row.get('Embedding_Path', ''),
                    'image_path': row.get('Image_Path', ''),
                    'created_date': row.get('Created_Date', '')
                }
            logger.info(f"Loaded {len(self.students_data)} students from CSV")
            return self.students_data
        except Exception as e:
            logger.error(f"Error loading students: {str(e)}")
            return {}
    
    def generate_embedding(self, image_path):
        """Generate face embedding from image"""
        for detector in self.detector_backends:
            try:
                logger.info(f"Generating embedding for {image_path} using {detector}")
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    detector_backend=detector,
                    enforce_detection=False,
                    align=True
                )
                
                if embedding_objs and len(embedding_objs) > 0:
                    logger.info(f"Successfully generated embedding using {detector}")
                    return embedding_objs[0]['embedding']
                    
            except Exception as e:
                logger.warning(f"Failed to generate embedding with {detector}: {str(e)}")
                continue
                
        logger.error(f"Failed to generate embedding with all detector backends for {image_path}")
        return None
    
    def save_embedding(self, student_id, embedding):
        """Save embedding to file"""
        try:
            embedding_path = os.path.join(EMBEDDINGS_FOLDER, f"{student_id}.npy")
            np.save(embedding_path, embedding)
            return embedding_path
        except Exception as e:
            logger.error(f"Error saving embedding: {str(e)}")
            return None
    
    def load_embedding(self, embedding_path):
        """Load embedding from file"""
        try:
            if os.path.exists(embedding_path):
                return np.load(embedding_path)
            return None
        except Exception as e:
            logger.error(f"Error loading embedding: {str(e)}")
            return None
    
    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings using DeepFace"""
        try:
            # Use DeepFace's distance calculation for consistency
            distance = DeepFace.dst.findCosineDistance(embedding1, embedding2)
            similarity = 1 - distance  # Convert distance to similarity score
            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0
    
    def recognize_face(self, image_path):
        """Recognize face in image and return matched student details"""
        try:
            logger.info(f"Starting face recognition for image: {image_path}")
            
            # Verify the image exists and is readable
            if not os.path.exists(image_path):
                logger.error(f"Image path does not exist: {image_path}")
                return None
                
            try:
                # Check if the image is valid and can be opened
                with Image.open(image_path) as img:
                    img_format = img.format
                    img_size = img.size
                    logger.info(f"Image details: Format={img_format}, Size={img_size}")
            except Exception as img_error:
                logger.error(f"Failed to open image: {image_path}, Error: {str(img_error)}")
                return None
            
            # Ensure the model is loaded before recognition
            self.load_model()
            
            # Reload student data to ensure it's fresh
            if not self.students_data:
                self.load_students()
                
            if len(self.students_data) == 0:
                logger.warning("No student data available for recognition")
                return None

            # Generate embedding for input image
            input_embedding = self.generate_embedding(image_path)
            if input_embedding is None:
                logger.warning(f"Could not generate embedding for image: {image_path}")
                return None

            best_match = None
            best_similarity = 0
            total_students = len(self.students_data)
            valid_embeddings = 0

            logger.info(f"Comparing against {total_students} students in database")

            # Compare with all stored students
            for student_id, student_info in self.students_data.items():
                embedding_path = student_info.get('embedding_path')
                if embedding_path and os.path.exists(embedding_path):
                    stored_embedding = self.load_embedding(embedding_path)
                    if stored_embedding is not None:
                        valid_embeddings += 1
                        similarity = self.cosine_similarity(input_embedding, stored_embedding)
                        logger.debug(f"Student {student_id}: similarity = {similarity:.4f}")

                        if similarity > best_similarity and similarity > self.threshold:
                            best_similarity = similarity
                            best_match = {
                                'student_id': student_id,
                                'name': student_info['name'],
                                'similarity': similarity,
                                'email': student_info.get('email', ''),
                                'phone': student_info.get('phone', ''),
                                'confidence': round(similarity * 100, 2)  # Convert to percentage for frontend
                            }
                            logger.info(f"New best match: {student_id} with confidence {similarity:.4f}")
                else:
                    logger.warning(f"Missing or invalid embedding for student {student_id}")

            logger.info(f"Recognition complete: {valid_embeddings}/{total_students} valid embeddings checked")
            if best_match:
                logger.info(f"Best match: {best_match['student_id']} ({best_match['name']}) with confidence {best_similarity:.4f}")
            else:
                logger.info(f"No match found above threshold {self.threshold}")

            return best_match

        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return None
    
    def group_photo_recognition(self, image_path):
        """Detect and recognize all faces in a group photo"""
        try:
            # Extract faces from the image
            logger.info(f"Starting face extraction from {image_path}")
            start_time = time.time()
            
            # Ensure the model is loaded
            self.load_model()
            
            # Reload student data
            self.load_students()
            
            faces = None
            # Try different detector backends for face extraction
            for detector in self.detector_backends:
                try:
                    logger.info(f"Trying face extraction with {detector}")
                    faces = DeepFace.extract_faces(
                        img_path=image_path,
                        detector_backend=detector,
                        enforce_detection=False,
                        align=True
                    )
                    if faces and len(faces) > 0:
                        logger.info(f"Successfully extracted {len(faces)} faces using {detector}")
                        break
                except Exception as e:
                    logger.warning(f"Face extraction failed with {detector}: {str(e)}")
                    continue
            
            if not faces:
                logger.warning(f"No faces detected in group photo {image_path}")
                return []
                
            logger.info(f"Detected {len(faces)} faces in group photo in {time.time() - start_time:.2f} seconds")
            
            results = []
            for i, face_obj in enumerate(faces):
                try:
                    # Get the face image
                    face = face_obj['face']
                    
                    # Save temporary face image
                    temp_face_path = os.path.join(TEMP_FOLDER, f"temp_face_{i}_{int(time.time())}.jpg")

                    # Convert face to proper format for saving
                    try:
                        # DeepFace returns faces as float arrays in range [0,1], need to convert to uint8 [0,255]
                        if face.dtype != np.uint8:
                            face_uint8 = (face * 255).astype(np.uint8)
                        else:
                            face_uint8 = face
                        
                        # Ensure the face has the right dimensions and format
                        if len(face_uint8.shape) == 3 and face_uint8.shape[2] == 3:
                            # Convert RGB to BGR for OpenCV
                            face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)
                            success = cv2.imwrite(temp_face_path, face_bgr)
                            if not success:
                                logger.warning(f"OpenCV failed to save face {i}, trying PIL")
                                # Fallback: use PIL to save
                                face_pil = Image.fromarray(face_uint8)
                                face_pil.save(temp_face_path)
                        else:
                            logger.warning(f"Unexpected face shape: {face_uint8.shape}, using PIL")
                            # Fallback: use PIL to save
                            face_pil = Image.fromarray(face_uint8)
                            face_pil.save(temp_face_path)
                            
                        logger.info(f"Successfully saved face {i} to {temp_face_path}")
                        
                    except Exception as save_error:
                        logger.error(f"Error saving face image {i}: {str(save_error)}")
                        continue

                    # Verify the face image was saved properly
                    if not os.path.exists(temp_face_path) or os.path.getsize(temp_face_path) == 0:
                        logger.error(f"Face image {i} was not saved properly")
                        continue
                    
                    # Recognize the face
                    match = self.recognize_face(temp_face_path)
                    results.append({
                        'face_index': i,
                        'match': match,
                        'face_image': os.path.basename(temp_face_path)
                    })
                    
                    # Clean up temp file
                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
                        
                except Exception as face_error:
                    logger.error(f"Error processing face {i}: {str(face_error)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in group photo recognition: {str(e)}")
            return []
            
    def extract_faces_for_bulk_recognition(self, image_path):
        """Extract and recognize all faces in an image for bulk attendance"""
        try:
            # Extract faces from the image
            start_time = time.time()
            logger.info(f"Starting face extraction from {image_path}")
            
            # Use fallback detectors for bulk processing
            faces = None
            for detector in self.detector_backends:
                try:
                    logger.info(f"Trying bulk face extraction with {detector}")
                    faces = DeepFace.extract_faces(
                        img_path=image_path,
                        detector_backend=detector,
                        enforce_detection=False,
                        align=True
                    )
                    if faces and len(faces) > 0:
                        logger.info(f"Successfully extracted {len(faces)} faces for bulk processing using {detector}")
                        break
                except Exception as e:
                    logger.warning(f"Bulk face extraction failed with {detector}: {str(e)}")
                    continue
            
            extraction_time = time.time() - start_time
            logger.info(f"Face extraction completed in {extraction_time:.2f} seconds. Found {len(faces)} faces.")
            
            if not faces:
                logger.warning("No faces detected in the uploaded image")
                return []
                
            # Process each detected face
            recognition_results = []
            for i, face_obj in enumerate(faces):
                try:
                    # Get the face image
                    face = face_obj['face']
                    
                    # Save temporary face image
                    temp_face_path = os.path.join(TEMP_FOLDER, f"bulk_face_{i}_{int(time.time())}.jpg")

                    # Convert face to proper format for saving
                    try:
                        # DeepFace returns faces as float arrays in range [0,1], need to convert to uint8 [0,255]
                        if face.dtype != np.uint8:
                            face_uint8 = (face * 255).astype(np.uint8)
                        else:
                            face_uint8 = face
                        
                        # Ensure the face has the right dimensions and format
                        if len(face_uint8.shape) == 3 and face_uint8.shape[2] == 3:
                            # Convert RGB to BGR for OpenCV
                            face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)
                            success = cv2.imwrite(temp_face_path, face_bgr)
                            if not success:
                                logger.warning(f"OpenCV failed to save bulk face {i}, trying PIL")
                                # Fallback: use PIL to save
                                face_pil = Image.fromarray(face_uint8)
                                face_pil.save(temp_face_path)
                        else:
                            logger.warning(f"Unexpected bulk face shape: {face_uint8.shape}, using PIL")
                            # Fallback: use PIL to save
                            face_pil = Image.fromarray(face_uint8)
                            face_pil.save(temp_face_path)
                            
                        logger.info(f"Successfully saved bulk face {i} to {temp_face_path}")
                        
                    except Exception as save_error:
                        logger.error(f"Error saving bulk face image {i}: {str(save_error)}")
                        continue

                    # Verify the face image was saved properly
                    if not os.path.exists(temp_face_path) or os.path.getsize(temp_face_path) == 0:
                        logger.error(f"Bulk face image {i} was not saved properly")
                        continue
                    
                    # Recognize the face
                    match = self.recognize_face(temp_face_path)
                    
                    if match:
                        recognition_results.append({
                            'student_id': match['student_id'],
                            'name': match['name'],
                            'confidence': match['confidence'],
                            'face_image': os.path.basename(temp_face_path)
                        })
                    else:
                        # Keep the face for reference even if not recognized
                        recognition_results.append({
                            'student_id': None,
                            'name': 'Unknown',
                            'confidence': 0,
                            'face_image': os.path.basename(temp_face_path)
                        })
                    
                    # Clean up temp file
                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
                        
                except Exception as face_error:
                    logger.error(f"Error processing bulk face {i}: {str(face_error)}")
                    continue
                
            recognition_time = time.time() - start_time - extraction_time
            logger.info(f"Face recognition completed in {recognition_time:.2f} seconds. Recognized {len([r for r in recognition_results if r['student_id']])} faces.")
            
            return recognition_results
            
        except Exception as e:
            logger.error(f"Error in bulk face recognition: {str(e)}")
            return []

# Initialize face recognition system
face_system = FaceRecognitionSystem()

# Add a test route to verify system functionality
@app.route('/test_face_recognition')
def test_face_recognition():
    """Test route to verify face recognition system"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    test_results = {
        'model_loaded': False,
        'students_loaded': False,
        'detector_backends': face_system.detector_backends,
        'current_config': {
            'model': face_system.model_name,
            'detector': face_system.detector_backend,
            'threshold': face_system.threshold
        },
        'student_count': 0,
        'embedding_files': []
    }
    
    try:
        # Test model loading
        test_results['model_loaded'] = face_system.load_model()
        
        # Test student loading
        face_system.load_students()
        test_results['students_loaded'] = len(face_system.students_data) > 0
        test_results['student_count'] = len(face_system.students_data)
        
        # Check embedding files
        if os.path.exists(EMBEDDINGS_FOLDER):
            embedding_files = [f for f in os.listdir(EMBEDDINGS_FOLDER) if f.endswith('.npy')]
            test_results['embedding_files'] = embedding_files[:10]  # Show first 10
            
    except Exception as e:
        test_results['error'] = str(e)
    
    return jsonify(test_results)

# Routes
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    # Simple login for demo - in production, use proper authentication
    username = request.form.get('username')
    password = request.form.get('password')
    
    logger.info(f"Login attempt - Username: '{username}', Password: '{password}'")
    logger.info(f"Form data: {request.form}")
    
    # For demo purposes, accept any non-empty credentials
    if username and password:
        logger.info("Login successful")
        session['logged_in'] = True
        session['username'] = username
        return redirect(url_for('dashboard'))
    else:
        logger.info("Login failed - empty credentials")
        flash('Please enter valid credentials', 'error')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    """Dashboard with statistics"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template(
        'dashboard.html', 
        stats=get_stats(), 
        current_user={'username': session.get('username', 'Admin'), 'is_authenticated': True}, 
        today=today
    )

@app.route('/students')
def students():
    """Students list page"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    face_system.load_students()
    return render_template(
        'students.html', 
        students=face_system.students_data, 
        current_user={'username': session.get('username', 'Admin'), 'is_authenticated': True}
    )

@app.route('/add_student')
def add_student():
    """Add student page"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template(
        'add_student.html', 
        current_user={'username': session.get('username', 'Admin'), 'is_authenticated': True}
    )

@app.route('/add_student', methods=['POST'])
def add_student_post():
    """Process add student form"""
    try:
        # Check authentication
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        
        student_id = request.form['student_id']
        name = request.form['name']
        email = request.form.get('email', '')
        phone = request.form.get('phone', '')
        
        # Handle file upload
        if 'face_image' not in request.files:
            flash('No photo uploaded', 'error')
            return redirect(url_for('add_student'))
        
        file = request.files['face_image']
        if file.filename == '':
            flash('No photo selected', 'error')
            return redirect(url_for('add_student'))
        
        # Save uploaded image
        filename = f"{student_id}_{file.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)
        
        # Generate face embedding
        embedding = face_system.generate_embedding(image_path)
        if embedding is None:
            flash('Could not detect face in the uploaded image', 'error')
            return redirect(url_for('add_student'))
        
        # Save embedding
        embedding_path = face_system.save_embedding(student_id, embedding)
        
        # Save to CSV
        file_exists = os.path.exists(STUDENT_DATA_CSV)
        with open(STUDENT_DATA_CSV, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['Student_ID', 'Name', 'Email', 'Phone', 'Embedding_Path', 'Image_Path', 'Created_Date'])
            writer.writerow([student_id, name, email, phone, embedding_path, image_path, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        # Reload students data
        face_system.load_students()
        
        flash('Student added successfully!', 'success')
        return redirect(url_for('students'))
        
    except Exception as e:
        logger.error(f"Error adding student: {str(e)}")
        flash(f'Error adding student: {str(e)}', 'error')
        return redirect(url_for('add_student'))

@app.route('/edit_student/<string:student_id>')
def edit_student(student_id):
    """Edit student page"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Load student data
    face_system.load_students()
    if student_id not in face_system.students_data:
        flash('Student not found', 'error')
        return redirect(url_for('students'))
    
    student_data = face_system.students_data[student_id]
    
    # Check if student has face embedding
    has_face_data = False
    if 'embedding_path' in student_data and os.path.exists(student_data['embedding_path']):
        has_face_data = True
    
    return render_template(
        'add_student.html',  # We'll reuse the add_student template for editing
        student=student_data,
        has_face_data=has_face_data,
        current_user={'username': session.get('username', 'Admin'), 'is_authenticated': True}
    )

@app.route('/edit_student/<string:student_id>', methods=['POST'])
def edit_student_post(student_id):
    """Process edit student form"""
    try:
        # Check authentication
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        
        # Load current student data
        face_system.load_students()
        if student_id not in face_system.students_data:
            flash('Student not found', 'error')
            return redirect(url_for('students'))
        
        current_student = face_system.students_data[student_id]
        
        # Get form data
        name = request.form['name']
        email = request.form.get('email', '')
        phone = request.form.get('phone', '')
        
        # Handle optional file upload for face image
        new_image_path = current_student.get('image_path', '')
        new_embedding_path = current_student.get('embedding_path', '')
        
        if 'face_image' in request.files and request.files['face_image'].filename:
            # Process new image
            file = request.files['face_image']
            
            # Save uploaded image
            filename = f"{student_id}_{file.filename}"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)
            
            # Generate face embedding
            embedding = face_system.generate_embedding(image_path)
            if embedding is None:
                flash('Could not detect face in the uploaded image', 'error')
                return redirect(url_for('edit_student', student_id=student_id))
            
            # Save embedding
            embedding_path = face_system.save_embedding(student_id, embedding)
            
            new_image_path = image_path
            new_embedding_path = embedding_path
        
        # Update CSV
        df = pd.read_csv(STUDENT_DATA_CSV)
        idx = df[df['Student_ID'] == student_id].index[0]
        df.at[idx, 'Name'] = name
        df.at[idx, 'Email'] = email
        df.at[idx, 'Phone'] = phone
        df.at[idx, 'Image_Path'] = new_image_path
        df.at[idx, 'Embedding_Path'] = new_embedding_path
        df.to_csv(STUDENT_DATA_CSV, index=False)
        
        # Reload students data
        face_system.load_students()
        
        flash('Student updated successfully!', 'success')
        return redirect(url_for('students'))
    except Exception as e:
        logger.error(f"Error updating student: {str(e)}")
        flash(f'Error updating student: {str(e)}', 'error')
        return redirect(url_for('edit_student', student_id=student_id))

@app.route('/delete_student/<string:student_id>')
def delete_student(student_id):
    """Delete a student"""
    try:
        # Check authentication
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        
        # Load current student data
        face_system.load_students()
        if student_id not in face_system.students_data:
            flash('Student not found', 'error')
            return redirect(url_for('students'))
        
        student_data = face_system.students_data[student_id]
        
        # Delete image and embedding files if they exist
        if 'image_path' in student_data and os.path.exists(student_data['image_path']):
            try:
                os.remove(student_data['image_path'])
            except Exception as e:
                logger.error(f"Error deleting image file: {str(e)}")
        
        if 'embedding_path' in student_data and os.path.exists(student_data['embedding_path']):
            try:
                os.remove(student_data['embedding_path'])
            except Exception as e:
                logger.error(f"Error deleting embedding file: {str(e)}")
        
        # Remove from CSV
        df = pd.read_csv(STUDENT_DATA_CSV)
        df = df[df['Student_ID'] != student_id]
        df.to_csv(STUDENT_DATA_CSV, index=False)
        
        # Reload students data
        face_system.load_students()
        
        flash('Student deleted successfully!', 'success')
        return redirect(url_for('students'))
    except Exception as e:
        logger.error(f"Error deleting student: {str(e)}")
        flash(f'Error deleting student: {str(e)}', 'error')
        return redirect(url_for('students'))

@app.route('/face_training')
def face_training():
    """Face training page"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Get list of students
    students = []
    if os.path.exists(STUDENT_DATA_CSV):
        with open(STUDENT_DATA_CSV, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            students = list(reader)
    
    # Get training stats
    stats = {
        'trained_faces_count': len(students),
        'total_training_images': 0,
        'average_accuracy': 95  # Placeholder value
    }
    
    # Count training images (this is a placeholder, actual implementation would depend on your system)
    embedding_folder = os.path.join('static', 'embeddings')
    if os.path.exists(embedding_folder):
        for _, _, files in os.walk(embedding_folder):
            stats['total_training_images'] += len([f for f in files if f.endswith('.npy')])
    
    # Create a current_user dict similar to other routes
    current_user = {'username': session.get('username', 'Admin'), 'is_authenticated': True}
    
    return render_template('face_training.html', 
                           students=students,
                           trained_faces_count=stats['trained_faces_count'],
                           total_training_images=stats['total_training_images'],
                           average_accuracy=stats['average_accuracy'],
                           current_user=current_user)

@app.route('/delete_face_training/<string:student_id>', methods=['DELETE'])
def delete_face_training(student_id):
    try:
        # Delete student's training data
        # This is a placeholder - implement based on your system's structure
        embedding_path = os.path.join('static', 'embeddings', f"{student_id}.npy")
        if os.path.exists(embedding_path):
            os.remove(embedding_path)
            return jsonify({"success": True, "message": f"Training data deleted for student {student_id}"})
        else:
            return jsonify({"success": False, "message": "Training data not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting training data: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500
        
@app.route('/train_existing_student', methods=['POST'])
def train_existing_student():
    try:
        student_id = request.form['student_id']
        
        # Check if student exists
        student_exists = False
        if os.path.exists(STUDENT_DATA_CSV):
            with open(STUDENT_DATA_CSV, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['Student_ID'] == student_id:
                        student_exists = True
                        break
        
        if not student_exists:
            return jsonify({"success": False, "message": "Student not found"}), 404
            
        # Handle uploaded training images
        if 'training_images' not in request.files:
            return jsonify({"success": False, "message": "No images uploaded"}), 400
            
        files = request.files.getlist('training_images')
        if not files or files[0].filename == '':
            return jsonify({"success": False, "message": "No images selected"}), 400
            
        # Process each uploaded image
        successful_images = 0
        failed_images = 0
        
        for file in files:
            try:
                # Save the uploaded image
                filename = f"{student_id}_training_{int(time.time())}_{file.filename}"
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(image_path)
                
                # Generate face embedding
                embedding = face_system.generate_embedding(image_path)
                if embedding is not None:
                    # Save embedding with a unique name to avoid overwriting
                    embedding_filename = f"{student_id}_training_{int(time.time())}.npy"
                    embedding_path = os.path.join('static', 'embeddings', embedding_filename)
                    np.save(embedding_path, embedding)
                    successful_images += 1
                else:
                    failed_images += 1
                    logger.warning(f"Could not detect face in {filename}")
            except Exception as e:
                failed_images += 1
                logger.error(f"Error processing training image: {str(e)}")
        
        # Return results
        return jsonify({
            "success": True, 
            "message": f"Training completed: {successful_images} images processed successfully, {failed_images} failed"
        })
        
    except Exception as e:
        logger.error(f"Error training existing student: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500
        
@app.route('/train_new_student', methods=['POST'])
def train_new_student():
    try:
        student_id = request.form['student_id']
        name = request.form['name']
        email = request.form.get('email', '')
        phone = request.form.get('phone', '')
        
        # Check if student ID already exists
        if os.path.exists(STUDENT_DATA_CSV):
            with open(STUDENT_DATA_CSV, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['Student_ID'] == student_id:
                        return jsonify({"success": False, "message": "Student ID already exists"}), 400
        
        # Handle uploaded training images
        if 'training_images' not in request.files:
            return jsonify({"success": False, "message": "No images uploaded"}), 400
            
        files = request.files.getlist('training_images')
        if not files or files[0].filename == '':
            return jsonify({"success": False, "message": "No images selected"}), 400
        
        # Process first image as main profile image
        main_file = files[0]
        main_filename = f"{student_id}_{main_file.filename}"
        main_image_path = os.path.join(UPLOAD_FOLDER, main_filename)
        main_file.save(main_image_path)
        
        # Generate face embedding from first image
        main_embedding = face_system.generate_embedding(main_image_path)
        if main_embedding is None:
            return jsonify({"success": False, "message": "Could not detect face in the primary image"}), 400
        
        # Save main embedding
        main_embedding_path = face_system.save_embedding(student_id, main_embedding)
        
        # Save to CSV
        file_exists = os.path.exists(STUDENT_DATA_CSV)
        with open(STUDENT_DATA_CSV, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['Student_ID', 'Name', 'Email', 'Phone', 'Embedding_Path', 'Image_Path', 'Created_Date'])
            writer.writerow([student_id, name, email, phone, main_embedding_path, main_image_path, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        # Process additional training images if any
        successful_images = 1  # Count the main image
        failed_images = 0
        
        for file in files[1:]:  # Skip the first file which was already processed
            try:
                # Save the uploaded image
                filename = f"{student_id}_training_{int(time.time())}_{file.filename}"
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(image_path)
                
                # Generate face embedding
                embedding = face_system.generate_embedding(image_path)
                if embedding is not None:
                    # Save embedding with a unique name
                    embedding_filename = f"{student_id}_training_{int(time.time())}.npy"
                    embedding_path = os.path.join('static', 'embeddings', embedding_filename)
                    np.save(embedding_path, embedding)
                    successful_images += 1
                else:
                    failed_images += 1
                    logger.warning(f"Could not detect face in {filename}")
            except Exception as e:
                failed_images += 1
                logger.error(f"Error processing training image: {str(e)}")
        
        # Reload students data
        face_system.load_students()
        
        # Return results
        return jsonify({
            "success": True, 
            "message": f"Student added successfully with {successful_images} training images. {failed_images} images failed."
        })
        
    except Exception as e:
        logger.error(f"Error adding new student with training: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/realtime_recognition')
def realtime_recognition():
    """Realtime recognition page"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('realtime_recognition.html', current_user={'username': session.get('username', 'Admin'), 'is_authenticated': True})

@app.route('/recognition')
def recognition():
    """Recognition page"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('recognition.html', current_user={'username': session.get('username', 'Admin'), 'is_authenticated': True})

@app.route('/recognize_upload', methods=['POST'])
def recognize_upload():
    """Process uploaded image for recognition"""
    try:
        # Check authentication
        if not session.get('logged_in'):
            return jsonify({'success': False, 'message': 'Not authenticated'})
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No image selected'})
        
        # Save uploaded image
        filename = f"recognize_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)
        
        # Ensure the image was saved properly
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            logger.error(f"Failed to save image properly at {image_path}")
            return jsonify({'success': False, 'message': 'Error saving uploaded image'})
        
        # Check if it's group photo recognition
        is_group = request.form.get('is_group', 'false').lower() == 'true'
        logger.info(f"Recognition type: {'Group' if is_group else 'Single'}")
        
        # Ensure students are loaded
        face_system.load_students()
        logger.info(f"Students loaded: {len(face_system.students_data)}")
        
        if is_group:
            # Group photo recognition
            logger.info("Starting group photo recognition")
            results = face_system.group_photo_recognition(image_path)
            logger.info(f"Group recognition results: {len(results)} faces processed")
            return jsonify({
                'success': True,
                'is_group': True,
                'results': results,
                'image_path': filename
            })
        else:
            # Single face recognition
            logger.info("Starting single face recognition")
            match = face_system.recognize_face(image_path)
            logger.info(f"Single recognition result: {match is not None}")
            
            if match:
                # Automatically mark attendance if enabled in config
                if AUTO_ATTENDANCE:
                    attendance_success = log_attendance(match['student_id'], match['name'])
                    match['attendance_marked'] = attendance_success
                    if attendance_success:
                        logger.info(f"Attendance automatically marked for {match['name']} ({match['student_id']})")
                    else:
                        logger.info(f"Attendance already marked today for {match['name']} ({match['student_id']})")

                return jsonify({
                    'success': True,
                    'is_group': False,
                    'match': match,
                    'image_path': filename
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No matching student found'
                })
                
    except Exception as e:
        logger.error(f"Error in recognition: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/recognize_realtime', methods=['POST'])
def recognize_realtime():
    """Process realtime webcam frames for recognition"""
    try:
        # Check authentication
        if not session.get('logged_in'):
            return jsonify({'success': False, 'message': 'Not authenticated'})
        
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        try:
            # Ensure proper base64 format (handle both with and without prefix)
            if ',' in image_data:
                image_data = image_data.split(',')[1]
                
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Save temporary image with exception handling
            temp_path = f"{TEMP_FOLDER}/realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image.save(temp_path)
            
            # Verify the image was saved properly
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                logger.error(f"Failed to save temporary image at {temp_path}")
                return jsonify({'success': False, 'message': 'Error processing webcam image'})
        except Exception as img_error:
            logger.error(f"Error processing webcam image data: {str(img_error)}")
            return jsonify({'success': False, 'message': f'Error processing image: {str(img_error)}'})
        
        # Recognize face
        match = face_system.recognize_face(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if match:
            # Log attendance
            log_attendance(match['student_id'], match['name'])
            return jsonify({
                'success': True,
                'data': [match]  # Return as array for compatibility with frontend
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No matching student found'
            })
            
    except Exception as e:
        logger.error(f"Error in realtime recognition: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    """Detect faces in an image"""
    try:
        # Check authentication
        if not session.get('logged_in'):
            return jsonify({'success': False, 'message': 'Not authenticated'})
        
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Save temporary image
        temp_path = f"{TEMP_FOLDER}/detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image.save(temp_path)
        
        # Use DeepFace to detect faces
        try:
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=face_system.detector_backend,
                enforce_detection=False
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'faces_detected': len(faces),
                'face_locations': []  # For now, just return count
            })
            
        except Exception as face_error:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return jsonify({
                'success': False,
                'faces_detected': 0,
                'error': str(face_error)
            })
            
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_route():
    """Mark attendance for a student"""
    try:
        # Check authentication
        if not session.get('logged_in'):
            return jsonify({'success': False, 'message': 'Not authenticated'})
        
        data = request.json
        student_id = data.get('student_id')
        student_name = data.get('name')
        
        if not student_id or not student_name:
            return jsonify({'success': False, 'message': 'Missing student information'})
        
        success = log_attendance(student_id, student_name)
        
        if success:
            return jsonify({'success': True, 'message': 'Attendance marked successfully'})
        else:
            return jsonify({'success': False, 'message': 'Attendance already marked for today'})
            
    except Exception as e:
        logger.error(f"Error marking attendance: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/mark_bulk_attendance', methods=['POST'])
def mark_bulk_attendance():
    """Mark attendance for multiple students"""
    try:
        # Check authentication
        if not session.get('logged_in'):
            return jsonify({'success': False, 'message': 'Not authenticated'})
        
        data = request.json
        student_ids = data.get('student_ids', [])
        
        if not student_ids:
            return jsonify({'success': False, 'message': 'No students provided'})
        
        # Get student names from IDs
        marked = []
        already_marked = []
        errors = []
        
        # Load students data to get names
        face_system.load_students()


        for student_id in student_ids:
            try:
                if student_id in face_system.students_data:
                    student_name = face_system.students_data[student_id]['name']
                    
                    # Mark attendance
                    success = log_attendance(student_id, student_name)
                    if success:
                        marked.append({
                            'student_id': student_id,
                            'name': student_name
                        })
                    else:
                        already_marked.append({
                            'student_id': student_id,
                            'name': student_name,
                            'reason': 'Attendance already marked for today'
                        })
                else:
                    errors.append({
                        'student_id': student_id,
                        'reason': 'Student not found in database'
                    })
            except Exception as e:
                logger.error(f"Error marking attendance for student {student_id}: {str(e)}")
                errors.append({
                    'student_id': student_id,
                    'reason': str(e)
                })
        
        return jsonify({
            'success': True,
            'marked_attendance': marked,
            'already_marked': already_marked,
            'errors': errors
        })
        
    except Exception as e:
        logger.error(f"Error in bulk attendance marking: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/bulk_attendance', methods=['POST'])
def bulk_attendance():
    """Process a group photo and mark attendance for all recognized students"""
    try:
        # Check authentication
        if not session.get('logged_in'):
            return jsonify({'success': False, 'message': 'Not authenticated'})
        
        # Check for image file - support both 'photo' and 'file' field names for compatibility
        file = None
        if 'photo' in request.files and request.files['photo'].filename != '':
            file = request.files['photo']
        elif 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
        
        if file is None:
            return jsonify({'success': False, 'message': 'No photo uploaded'})
        # Save uploaded image
        filename = f"group_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)
        
        # Extract faces and recognize students
        start_time = time.time()
        recognition_results = face_system.extract_faces_for_bulk_recognition(image_path)
        
        if not recognition_results:
            return jsonify({
                'success': False, 
                'message': 'No faces detected in the uploaded image'
            })
        
        # Filter results to get valid student IDs with good confidence
        min_confidence = 70  # Minimum confidence threshold (%)
        valid_students = [r for r in recognition_results if r['student_id'] and r['confidence'] >= min_confidence]
        student_ids = [s['student_id'] for s in valid_students]
        
        # Mark attendance for recognized students
        marked = []
        already_marked = []
        errors = []
        
        # Load students data to get names
        face_system.load_students()
        
        for student_id in student_ids:
            try:
                if student_id in face_system.students_data:
                    student_name = face_system.students_data[student_id]['name']
                    
                    # Mark attendance
                    success = log_attendance(student_id, student_name)
                    if success:
                        marked.append({
                            'student_id': student_id,
                            'name': student_name
                        })
                    else:
                        already_marked.append({
                            'student_id': student_id,
                            'name': student_name,
                            'reason': 'Already marked for today'
                        })
                else:
                    errors.append({
                        'student_id': student_id,
                        'reason': 'Student not found in database'
                    })
            except Exception as e:
                logger.error(f"Error marking attendance for student {student_id}: {str(e)}")
                errors.append({
                    'student_id': student_id,
                    'reason': str(e)
                })
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Extract any location data from image if requested
        extract_location = request.form.get('extract_location', 'true').lower() == 'true'
        location_data = {
            'has_location': False,
            'latitude': None,
            'longitude': None,
            'address': None
        }
        
        # Return results
        return jsonify({
            'success': True,
            'data': {
                'total_faces': len(recognition_results),
                'recognized_faces': len(valid_students),
                'marked_attendance': marked,
                'already_marked': already_marked,
                'errors': errors,
                'processing_time': processing_time,
                'location_data': location_data
            }
        })
        
    except Exception as e:
        logger.error(f"Error in bulk attendance processing: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})



@app.route('/attendance')
def attendance():
    """Attendance page"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    attendance_data = load_attendance_data()
    
    # Get statistics for the attendance page
    attendance_stats = get_stats()
    
    # Rename keys to match what the template expects
    stats = {
        'total_today': attendance_stats['total_attendance_today'],
        'total_week': attendance_stats['total_attendance_week'],
        'total_month': attendance_stats['total_attendance_month'],
        'avg_daily': round(attendance_stats['total_attendance_month'] / 30, 1) if attendance_stats['total_attendance_month'] > 0 else 0
    }
    
    return render_template('attendance.html', attendance=attendance_data, stats=stats, current_user={'username': session.get('username', 'Admin'), 'is_authenticated': True})

@app.route('/attendance_today')
def attendance_today():
    """Today's attendance page"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Load all attendance data and filter for today
    all_attendance = load_attendance_data()
    today = date.today().strftime('%Y-%m-%d')

    # Filter attendance for today only
    today_attendance = [record for record in all_attendance if record.get('date') == today]
    
    # Get statistics for the attendance page
    attendance_stats = get_stats()
    
    # Rename keys to match what the template expects
    stats = {
        'total_today': attendance_stats['total_attendance_today'],
        'total_week': attendance_stats['total_attendance_week'],
        'total_month': attendance_stats['total_attendance_month'],
        'avg_daily': round(attendance_stats['total_attendance_month'] / 30, 1) if attendance_stats['total_attendance_month'] > 0 else 0
    }

    return render_template('attendance.html', attendance=today_attendance, stats=stats, current_user={'username': session.get('username', 'Admin'), 'is_authenticated': True})

@app.route('/export_data/<data_type>')
def export_data(data_type):
    """Export data as CSV"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        if data_type == 'attendance':
            # Export attendance data
            attendance_data = load_attendance_data()
            if not attendance_data:
                return jsonify({'success': False, 'message': 'No attendance data to export'})

            # Create CSV response
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['Student_ID', 'Name', 'Date', 'Time'])
            writer.writeheader()
            for record in attendance_data:
                writer.writerow({
                    'Student_ID': record.get('Student_ID', ''),
                    'Name': record.get('Name', ''),
                    'Date': record.get('Date', ''),
                    'Time': record.get('Time', '')
                })

            # Create response
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename=attendance_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            return response

        elif data_type == 'students':
            # Export student data
            face_system.load_students()
            if not face_system.students_data:
                return jsonify({'success': False, 'message': 'No student data to export'})

            # Create CSV response
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['Student_ID', 'Name', 'Email', 'Phone'])
            writer.writeheader()
            for student in face_system.students_data:
                writer.writerow({
                    'Student_ID': student.get('Student_ID', ''),
                    'Name': student.get('Name', ''),
                    'Email': student.get('Email', ''),
                    'Phone': student.get('Phone', '')
                })

            # Create response
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename=students_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            return response

        else:
            return jsonify({'success': False, 'message': 'Invalid data type'})

    except Exception as e:
        logger.error(f"Error exporting {data_type} data: {str(e)}")
        return jsonify({'success': False, 'message': f'Export failed: {str(e)}'})

@app.route('/student_photo/<student_id>')
def student_photo(student_id):
    """Serve student photo"""
    # Check authentication
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        # Look for student photo in the embeddings directory
        photo_path = os.path.join(EMBEDDINGS_FOLDER, f"{student_id}.jpg")

        if os.path.exists(photo_path):
            return send_from_directory(EMBEDDINGS_FOLDER, f"{student_id}.jpg")
        else:
            # Return a default placeholder image or 404
            # For now, return a simple response indicating no photo
            return '', 404

    except Exception as e:
        logger.error(f"Error serving student photo for {student_id}: {str(e)}")
        return '', 404

def log_attendance(student_id, student_name):
    """Log student attendance"""
    try:
        if not student_id:
            logger.error("Cannot log attendance: Empty student ID")
            return False
            
        if not student_name:
            logger.error(f"Cannot log attendance: Empty student name for ID {student_id}")
            return False
        
        today = date.today().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')
        
        logger.info(f"Attempting to log attendance for {student_name} (ID: {student_id}) on {today}")

        # Ensure attendance CSV exists and has proper headers
        file_exists = os.path.exists(ATTENDANCE_CSV)
        if not file_exists:
            try:
                with open(ATTENDANCE_CSV, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Student_ID', 'Name', 'Date', 'Time'])
                logger.info(f"Created attendance CSV file: {ATTENDANCE_CSV}")
            except Exception as csv_error:
                logger.error(f"Failed to create attendance CSV file: {str(csv_error)}")
                return False

        # Check if already logged today
        try:
            df = pd.read_csv(ATTENDANCE_CSV)
            if not df.empty:
                existing = df[(df['Student_ID'] == student_id) & (df['Date'] == today)]
                if not existing.empty:
                    logger.info(f"Attendance already logged for student {student_id} on {today}")
                    return False  # Already logged today
        except (pd.errors.EmptyDataError, FileNotFoundError):
            # File is empty or doesn't exist, continue with logging
            logger.info("No existing attendance records found, creating new record")
        except Exception as read_error:
            logger.error(f"Error checking existing attendance: {str(read_error)}")
            # Continue anyway to try to mark attendance

        # Log new attendance
        try:
            with open(ATTENDANCE_CSV, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([student_id, student_name, today, current_time])
            logger.info(f"Attendance successfully logged for student {student_id} ({student_name}) at {current_time}")
            return True
        except Exception as write_error:
            logger.error(f"Error writing attendance record: {str(write_error)}")
            return False
        
    except Exception as e:
        logger.error(f"Error logging attendance: {str(e)}")
        return False

def load_attendance_data():
    """Load attendance data from CSV"""
    if not os.path.exists(ATTENDANCE_CSV):
        return []
    
    try:
        df = pd.read_csv(ATTENDANCE_CSV)
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error loading attendance data: {str(e)}")
        return []

def get_stats():
    """Get dashboard statistics"""
    face_system.load_students()
    total_students = len(face_system.students_data)
    
    # Get attendance stats
    today = date.today().strftime('%Y-%m-%d')
    attendance_today = 0
    attendance_week = 0
    attendance_month = 0
    
    if os.path.exists(ATTENDANCE_CSV):
        try:
            df = pd.read_csv(ATTENDANCE_CSV)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Today's attendance
            attendance_today = len(df[df['Date'].dt.date == date.today()])
            
            # This week's attendance
            week_start = date.today() - pd.Timedelta(days=date.today().weekday())
            attendance_week = len(df[df['Date'].dt.date >= week_start])
            
            # This month's attendance
            month_start = date.today().replace(day=1)
            attendance_month = len(df[df['Date'].dt.date >= month_start])
            
        except Exception as e:
            logger.error(f"Error calculating stats: {str(e)}")
    
    return {
        'total_students': total_students,
        'total_attendance_today': attendance_today,
        'total_attendance_week': attendance_week,
        'total_attendance_month': attendance_month
    }

if __name__ == '__main__':
    # Load students data on startup
    face_system.load_students()
    logger.info(f"Starting Face Recognition System...")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Detector: {DETECTOR_BACKEND}")
    logger.info(f"Threshold: {RECOGNITION_THRESHOLD}")
    app.run(debug=DEBUG_MODE, host=HOST, port=PORT)