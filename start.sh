#!/bin/bash
echo "Starting Face Recognition System..."
echo

# Check if virtual environment exists
if [ ! -d "face_recognition_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv face_recognition_env
    echo
fi

# Activate virtual environment
echo "Activating virtual environment..."
source face_recognition_env/bin/activate

# Check if requirements are installed
echo "Checking dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p static/uploads
mkdir -p static/embeddings
mkdir -p static/temp

echo
echo "Starting Flask application..."
echo "Open your browser and go to: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo

python app.py