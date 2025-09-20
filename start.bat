@echo off
echo Starting Face Recognition System...
echo.

REM Check if virtual environment exists
if not exist "face_recognition_env" (
    echo Creating virtual environment...
    python -m venv face_recognition_env
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call face_recognition_env\Scripts\activate

REM Check if requirements are installed
echo Checking dependencies...
pip install -r requirements.txt

REM Create necessary directories
if not exist "static\uploads" mkdir "static\uploads"
if not exist "static\embeddings" mkdir "static\embeddings"
if not exist "static\temp" mkdir "static\temp"

echo.
echo Starting Flask application...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause