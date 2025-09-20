# Face Recognition System Configuration
# Modify these settings according to your needs

# Model Configuration
MODEL_NAME = "Facenet"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace
DETECTOR_BACKEND = "retinaface"  # Options: opencv, mtcnn, ssd, dlib, retinaface, mediapipe (retinaface is more accurate)
DISTANCE_METRIC = "cosine"  # Options: cosine, euclidean, euclidean_l2
RECOGNITION_THRESHOLD = 0.4  # 0.0 to 1.0, lower = more lenient (improved for better recognition)

# Application Settings
SECRET_KEY = "your-secret-key-change-in-production"
DEBUG_MODE = True
HOST = "0.0.0.0"
PORT = 5000

# File Paths
UPLOAD_FOLDER = "static/uploads"
EMBEDDINGS_FOLDER = "static/embeddings"
TEMP_FOLDER = "static/temp"
STUDENT_DATA_CSV = "students.csv"
ATTENDANCE_CSV = "attendance.csv"

# Camera Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Recognition Settings
REALTIME_RECOGNITION_INTERVAL = 1000  # milliseconds
FACE_DETECTION_INTERVAL = 500  # milliseconds
AUTO_ATTENDANCE = True  # Automatically mark attendance when face is recognized

# UI Settings
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Performance Settings
MAX_WORKERS = 4  # For parallel processing
CACHE_EMBEDDINGS = True  # Keep embeddings in memory for faster processing