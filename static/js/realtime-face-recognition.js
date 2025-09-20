// Advanced Real-time Face Recognition
class RealtimeFaceRecognition {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.context = null;
        this.stream = null;
        this.isRecognitionActive = false;
        this.detectionInterval = null;
        this.recognitionInterval = null;
        this.resultDisplayed = false;
        this.noFaceTimer = null;
        this.attendanceMarkedTimestamps = new Map();
        
        this.init();
    }
    
    init() {
        this.setupDOM();
        this.setupEventListeners();
    }
    
    setupDOM() {
        // Initialize video element
        this.video = document.getElementById('cameraPreview');
        
        // Initialize canvas for frame capture
        this.canvas = document.createElement('canvas');
        this.context = this.canvas.getContext('2d');
        
        // Initialize streaming status
        this.updateStreamingStatus('ready');
    }
    
    setupEventListeners() {
        // Camera control buttons
        const startCameraBtn = document.getElementById('startCamera');
        const stopCameraBtn = document.getElementById('stopCamera');
        const startRecognitionBtn = document.getElementById('startRecognition');
        const stopRecognitionBtn = document.getElementById('stopRecognition');
        
        if (startCameraBtn) {
            startCameraBtn.addEventListener('click', this.startCamera.bind(this));
        }
        
        if (stopCameraBtn) {
            stopCameraBtn.addEventListener('click', this.stopCamera.bind(this));
        }
        
        if (startRecognitionBtn) {
            startRecognitionBtn.addEventListener('click', this.startRealtimeRecognition.bind(this));
        }
        
        if (stopRecognitionBtn) {
            stopRecognitionBtn.addEventListener('click', this.stopRealtimeRecognition.bind(this));
        }
    }
    
    async startCamera() {
        try {
            // Check if browser supports getUserMedia
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                this.showAlert('Your browser does not support camera access. Please use Chrome, Firefox, or Edge.', 'error');
                return;
            }
            
            // Reset any previous stream
            if (this.stream) {
                this.stopCamera();
            }
            
            // Try to get camera access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            });
            
            // Connect stream to video element
            if (this.video) {
                this.video.srcObject = this.stream;
                
                // Update UI when video is ready
                this.video.onloadedmetadata = () => {
                    // Show stop button, hide start button
                    const startBtn = document.getElementById('startCamera');
                    const stopBtn = document.getElementById('stopCamera');
                    const startRecBtn = document.getElementById('startRecognition');
                    
                    if (startBtn) startBtn.style.display = 'none';
                    if (stopBtn) stopBtn.style.display = 'block';
                    if (startRecBtn) startRecBtn.disabled = false;
                    
                    // Update status
                    this.updateStreamingStatus('ready');
                    
                    this.showAlert('Camera started successfully', 'success');
                };
            }
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showAlert('Failed to access camera. Please check permissions.', 'error');
        }
    }
    
    stopCamera() {
        // Stop any active recognition
        if (this.isRecognitionActive) {
            this.stopRealtimeRecognition();
        }
        
        // Stop all tracks in the stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        // Clear video source
        if (this.video) {
            this.video.srcObject = null;
        }
        
        // Update UI
        const startBtn = document.getElementById('startCamera');
        const stopBtn = document.getElementById('stopCamera');
        const startRecBtn = document.getElementById('startRecognition');
        
        if (startBtn) startBtn.style.display = 'block';
        if (stopBtn) stopBtn.style.display = 'none';
        if (startRecBtn) startRecBtn.disabled = true;
        
        this.showAlert('Camera stopped', 'info');
    }
    
    updateStreamingStatus(status, details = '') {
        const streamingStatus = document.getElementById('streamingStatus');
        if (!streamingStatus) return;
        
        // Remove all status classes first
        streamingStatus.classList.remove('scanning', 'detected', 'recognizing', 'unknown');
        
        let statusText = '';
        let iconClass = 'fa-video';
        
        switch(status) {
            case 'ready':
                statusText = 'Camera Ready';
                iconClass = 'fa-video';
                break;
            case 'scanning':
                statusText = 'Scanning for Faces...';
                iconClass = 'fa-search';
                streamingStatus.classList.add('scanning');
                break;
            case 'detected':
                statusText = details || 'Face Detected';
                iconClass = 'fa-user';
                streamingStatus.classList.add('detected');
                break;
            case 'recognizing':
                statusText = details || 'Recognizing Face...';
                iconClass = 'fa-spinner fa-spin';
                streamingStatus.classList.add('recognizing');
                break;
            case 'recognized':
                statusText = details || 'Face Recognized';
                iconClass = 'fa-check-circle';
                streamingStatus.classList.add('recognizing');
                break;
            case 'unknown':
                statusText = 'Unknown Face Detected';
                iconClass = 'fa-question-circle';
                streamingStatus.classList.add('unknown');
                break;
        }
        
        // Update the status text and icon
        streamingStatus.querySelector('.streaming-text').textContent = statusText;
        streamingStatus.querySelector('.streaming-icon i').className = `fas ${iconClass}`;
    }
    
    startRealtimeRecognition() {
        if (!this.video || !this.stream) {
            this.showAlert('Camera not active', 'error');
            return;
        }

        this.isRecognitionActive = true;
        this.updateRecognitionButtonsState(true);
        this.updateStreamingStatus('scanning');
        
        // Start continuous frame processing for detection
        this.detectionInterval = setInterval(() => {
            this.detectAndRecognizeFaces();
        }, 300); // Scan for faces every 300ms for smoother experience
        
        // Update UI
        document.getElementById('realTimeResults').innerHTML = 
            '<div class="scanning-animation"><i class="fas fa-search"></i><p>Scanning for faces...</p></div>';
        
        this.showAlert('Real-time recognition started', 'success');
    }

    stopRealtimeRecognition() {
        this.isRecognitionActive = false;
        this.updateRecognitionButtonsState(false);
        this.updateStreamingStatus('ready');
        
        // Clear all intervals
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        
        if (this.recognitionInterval) {
            clearInterval(this.recognitionInterval);
            this.recognitionInterval = null;
        }

        this.clearRecognitionResults();
        this.showAlert('Real-time recognition stopped', 'info');
    }
    
    updateRecognitionButtonsState(isActive) {
        // Update button visibility
        const startBtn = document.getElementById('startRecognition');
        const stopBtn = document.getElementById('stopRecognition');
        
        if (startBtn) {
            startBtn.style.display = isActive ? 'none' : 'block';
        }
        
        if (stopBtn) {
            stopBtn.style.display = isActive ? 'block' : 'none';
        }
    }
    
    async detectAndRecognizeFaces() {
        if (!this.video || !this.stream || !this.isRecognitionActive) return;
        
        try {
            // Update canvas with current video frame
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.context.drawImage(this.video, 0, 0);
            
            // Send frame data for detection at lower quality for better performance
            const imageData = FaceRecognitionMethods.captureFrame(this.video, 0.3);
            
            // First step: Detect faces (lightweight call)
            this.updateStreamingStatus('scanning');
            
            // Use the FaceRecognitionMethods utility class for detection
            const detectResult = await FaceRecognitionMethods.detectFaces(imageData);
            
            // Update camera container to show detection status
            const cameraContainer = document.querySelector('.camera-container');
            if (cameraContainer) {
                if (detectResult.success && detectResult.faces_detected > 0) {
                    cameraContainer.classList.add('detection-active');
                } else {
                    cameraContainer.classList.remove('detection-active');
                }
            }
            
            // Second step: If faces detected, perform recognition
            if (detectResult.success && detectResult.faces_detected > 0) {
                // Visual feedback for detection
                this.showFaceDetectedEffect();
                
                // Update status
                this.updateStreamingStatus('detected', `${detectResult.faces_detected} face(s) detected`);
                
                // Now perform full recognition with the same frame
                await this.recognizeFaces(imageData, detectResult.faces_detected);
            } else {
                // No faces detected, update UI accordingly
                this.updateStreamingStatus('scanning');
                
                // Clear any old results if no faces are visible for 3 seconds
                if (!this.noFaceTimer) {
                    this.noFaceTimer = setTimeout(() => {
                        if (this.isRecognitionActive) {
                            // Only clear results if we're still active
                            const resultsContainer = document.getElementById('realTimeResults');
                            if (resultsContainer) {
                                resultsContainer.innerHTML = 
                                    '<div class="scanning-animation"><i class="fas fa-search"></i><p>Scanning for faces...</p></div>';
                            }
                        }
                        this.noFaceTimer = null;
                    }, 3000);
                }
            }
        } catch (error) {
            console.error('Error in detection/recognition cycle:', error);
            // Don't update UI on errors to avoid flickering
        }
    }
    
    showFaceDetectedEffect() {
        // Add visual feedback when face is detected
        const overlay = document.getElementById('cameraOverlay');
        if (overlay) {
            overlay.innerHTML = `<div class="face-detected-flash"></div>`;
            setTimeout(() => {
                if (overlay) overlay.innerHTML = '';
            }, 800);
        }
    }
    
    async recognizeFaces(imageData, faceCount) {
        try {
            this.updateStreamingStatus('recognizing');
            
            // Use the FaceRecognitionMethods utility class for recognition
            const result = await FaceRecognitionMethods.recognizeFaces(imageData);
            
            // Clear the no-face timer since we have results
            if (this.noFaceTimer) {
                clearTimeout(this.noFaceTimer);
                this.noFaceTimer = null;
            }
            
            if (result.success) {
                if (result.data && result.data.length > 0) {
                    // Display recognized faces
                    this.displayRealtimeResults(result.data);
                    
                    // Update status with recognition info
                    this.updateStreamingStatus('recognized', `${result.data.length} face(s) recognized`);
                    
                    // Mark attendance automatically for each recognized face
                    this.autoMarkAttendance(result.data);
                } else if (result.faces_detected > 0) {
                    // Faces detected but not recognized
                    this.displayUnknownFaces(result.faces_detected);
                    this.updateStreamingStatus('unknown');
                }
            }
        } catch (error) {
            console.error('Error in face recognition:', error);
        }
    }
    
    displayRealtimeResults(recognizedFaces) {
        const container = document.getElementById('realTimeResults');
        if (!container) return;
        
        // Use ResultDisplayMethods utility class to display results
        this.resultDisplayed = ResultDisplayMethods.displayRealtimeResults(recognizedFaces, container, this.resultDisplayed);
        
        // Set up callback for attendance marking
        window.markAttendanceCallback = (face) => {
            this.markAttendanceForRecognizedFace(face);
        };
        
        // For compatibility with old code, still process each face for additional customization if needed
        recognizedFaces.forEach(face => {
            // Generate a unique ID for this result based on student ID and timestamp
            const resultId = `result-${face.student_id}-${Date.now()}`;
            
            // Check if we already displayed this person recently (in the last 5 seconds)
            const existingResult = container.querySelector(`[data-student-id="${face.student_id}"]`);
            
            if (existingResult) {
                // Just update the confidence if it's higher
                const confidenceBar = existingResult.querySelector('.confidence-bar');
                const confidenceText = existingResult.querySelector('.confidence-text');
                
                if (confidenceBar && confidenceText && face.confidence > parseFloat(existingResult.dataset.confidence)) {
                    // Update confidence display
                    confidenceBar.style.width = `${face.confidence}%`;
                    confidenceText.textContent = `${Math.round(face.confidence)}% match`;
                    
                    // Update the stored confidence value
                    existingResult.dataset.confidence = face.confidence;
                    
                    // Update the confidence class
                    this.updateConfidenceClass(existingResult, face.confidence);
                }
                
                // Update timestamp
                const timestamp = existingResult.querySelector('.timestamp');
                if (timestamp) {
                    timestamp.textContent = `Updated: ${this.getCurrentTime()}`;
                }
            } else {
                // Create new result card
                const resultHTML = this.createResultCardHTML(face);
                
                // Create container for the new result
                const resultContainer = document.createElement('div');
                resultContainer.className = 'realtime-result-container';
                resultContainer.innerHTML = resultHTML;
                
                // Get the actual card element
                const resultCard = resultContainer.querySelector('.realtime-result-card');
                
                // Store student ID and confidence for future updates
                resultCard.dataset.studentId = face.student_id;
                resultCard.dataset.confidence = face.confidence;
                
                // Add the confidence class
                this.updateConfidenceClass(resultCard, face.confidence);
                
                // Prepend to show newest results at the top
                container.prepend(resultContainer);
                
                // Apply entrance animation
                setTimeout(() => {
                    resultContainer.style.opacity = '1';
                    resultContainer.style.transform = 'translateY(0)';
                }, 10);
                
                // Mark attendance automatically
                this.markAttendanceForRecognizedFace(face);
            }
        });
    }
    
    createResultCardHTML(face) {
        // Format confidence for display
        const confidence = Math.round(face.confidence);
        
        // Get current time
        const currentTime = this.getCurrentTime();
        
        // Determine confidence level for visual styling
        let confidenceClass = 'low-confidence';
        let confidenceBadge = 'badge-danger';
        let confidenceLabel = 'Low Match';
        
        if (confidence >= 85) {
            confidenceClass = 'high-confidence';
            confidenceBadge = 'badge-success';
            confidenceLabel = 'Strong Match';
        } else if (confidence >= 70) {
            confidenceClass = 'medium-confidence';
            confidenceBadge = 'badge-warning';
            confidenceLabel = 'Possible Match';
        }
        
        // Build the HTML for the result card
        return `
            <div class="realtime-result-card ${confidenceClass}">
                <div class="result-photo" style="background-image: url('/student_photo/${face.student_id}')"></div>
                <div class="result-info">
                    <div class="student-name">${face.name}</div>
                    <div class="student-details">
                        ${face.student_id} | ${face.department || 'N/A'} | ${face.year || 'N/A'}
                    </div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" style="width: ${confidence}%"></div>
                    </div>
                    <div class="confidence-text">${confidence}% match</div>
                </div>
                <div class="badge ${confidenceBadge}">${confidenceLabel}</div>
                <div class="timestamp">Recognized: ${currentTime}</div>
            </div>
        `;
    }
    
    displayUnknownFaces(faceCount) {
        const container = document.getElementById('realTimeResults');
        if (!container) return;
        
        // Use ResultDisplayMethods utility class to display unknown faces
        const displayed = ResultDisplayMethods.displayUnknownFaces(faceCount, container);
        if (displayed) {
            this.resultDisplayed = true;
        }
    }
    
    updateConfidenceClass(element, confidence) {
        // Use ResultDisplayMethods utility class to update confidence class
        ResultDisplayMethods.updateConfidenceClass(element, confidence);
    }
    
    getCurrentTime() {
        // Use ResultDisplayMethods utility class to get formatted time
        return ResultDisplayMethods.getCurrentTime();
    }
    
    clearRecognitionResults() {
        // Use ResultDisplayMethods utility class to clear results
        if (ResultDisplayMethods.clearRecognitionResults()) {
            this.resultDisplayed = false;
        }
    }
    
    // Auto attendance marking for recognized faces
    autoMarkAttendance(recognizedFaces) {
        // Filter for high-confidence matches
        const reliableFaces = recognizedFaces.filter(face => face.confidence >= 80);
        
        // Skip if no reliable matches
        if (reliableFaces.length === 0) return;
        
        // Get the student IDs
        const studentIds = reliableFaces.map(face => face.student_id);
        
        // Check if we've already marked attendance for these students recently
        const now = Date.now();
        const recentlyMarked = studentIds.filter(id => {
            const lastMarked = this.attendanceMarkedTimestamps.get(id) || 0;
            return (now - lastMarked) < 60000; // Less than 1 minute ago
        });
        
        // Only process students we haven't recently marked
        const newAttendances = studentIds.filter(id => !recentlyMarked.includes(id));
        
        if (newAttendances.length === 0) return;
        
        // Mark attendance
        this.markAttendanceForStudents(newAttendances);
        
        // Update timestamps to prevent rapid repeated attendance marking
        newAttendances.forEach(id => {
            this.attendanceMarkedTimestamps.set(id, now);
        });
    }
    
    async markAttendanceForStudents(studentIds) {
        // Skip if empty
        if (!studentIds.length) return;
        
        console.log('Automatically marking attendance for:', studentIds);
        
        try {
            // Use FaceRecognitionMethods utility class for attendance marking
            const result = await FaceRecognitionMethods.markAttendance(studentIds);
            
            if (result.success) {
                console.log(`Attendance marked for ${result.marked_count} students`);
                // Show a subtle notification
                ResultDisplayMethods.showAttendanceMarkedNotification(result.marked_count);
            }
        } catch (error) {
            console.error('Error marking attendance:', error);
        }
    }
    
    markAttendanceForRecognizedFace(face) {
        // Only mark attendance for high confidence matches
        if (face.confidence < 80) return;
        
        // Check if we've already marked attendance for this student recently
        const now = Date.now();
        const lastMarked = this.attendanceMarkedTimestamps.get(face.student_id) || 0;
        if ((now - lastMarked) < 60000) { // Less than 1 minute ago
            return;
        }
        
        // Mark attendance
        this.markAttendanceForStudents([face.student_id]);
        
        // Update timestamp
        this.attendanceMarkedTimestamps.set(face.student_id, now);
    }
    
    showAttendanceMarkedNotification(count) {
        // Use ResultDisplayMethods utility class to show notification
        ResultDisplayMethods.showAttendanceMarkedNotification(count);
    }
    
    showAlert(message, type) {
        console.log(`Alert (${type}): ${message}`);
        
        // You can replace this with your preferred alert system
        const alertContainer = document.createElement('div');
        alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
        alertContainer.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        const alertArea = document.querySelector('.alert-container');
        if (alertArea) {
            alertArea.appendChild(alertContainer);
            
            // Auto dismiss after 5 seconds
            setTimeout(() => {
                alertContainer.classList.remove('show');
                setTimeout(() => {
                    if (alertContainer.parentNode === alertArea) {
                        alertArea.removeChild(alertContainer);
                    }
                }, 300);
            }, 5000);
        }
    }
}

// Initialize the real-time face recognition when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on the recognition page
    if (document.getElementById('cameraPreview')) {
        window.faceRecognition = new RealtimeFaceRecognition();
    }
});

// Add notification styles to the document
const style = document.createElement('style');
style.textContent = `
.attendance-marked-notification {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(15, 23, 42, 0.9);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    z-index: 9999;
    transform: translateY(100px);
    opacity: 0;
    transition: all 0.3s ease;
}

.attendance-marked-notification.show {
    transform: translateY(0);
    opacity: 1;
}

.notification-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: rgba(52, 211, 153, 0.2);
    color: #34d399;
    display: flex;
    align-items: center;
    justify-content: center;
}

.scanning-animation {
    text-align: center;
    padding: 30px;
    color: rgba(255, 255, 255, 0.7);
}

.scanning-animation i {
    font-size: 32px;
    color: #38bdf8;
    margin-bottom: 15px;
    display: block;
    animation: pulse 1.5s infinite;
}
`;
document.head.appendChild(style);