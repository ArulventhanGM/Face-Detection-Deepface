/**
 * Face Recognition Methods
 * This file contains methods for face detection and recognition
 */

class FaceRecognitionMethods {
    /**
     * Detects faces in the provided image
     * @param {string} imageData - Base64 encoded image data
     * @returns {Promise} - Detection results
     */
    static async detectFaces(imageData) {
        try {
            const response = await fetch('/detect_faces', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error detecting faces:', error);
            return { success: false, error: error.message };
        }
    }
    
    /**
     * Recognizes faces in the provided image
     * @param {string} imageData - Base64 encoded image data
     * @returns {Promise} - Recognition results
     */
    static async recognizeFaces(imageData) {
        try {
            const response = await fetch('/recognize_realtime', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error recognizing faces:', error);
            return { success: false, error: error.message };
        }
    }
    
    /**
     * Captures a frame from video element
     * @param {HTMLVideoElement} videoElement - Video element to capture from
     * @param {number} quality - JPEG quality (0-1)
     * @returns {string} - Base64 encoded image data
     */
    static captureFrame(videoElement, quality = 0.7) {
        // Create canvas to capture frame
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        
        // Draw current frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0);
        
        // Convert to base64 JPEG
        return canvas.toDataURL('image/jpeg', quality);
    }
    
    /**
     * Calculate face detection area indicators
     * @param {Object} detectionResult - Face detection result from API
     * @param {HTMLElement} videoContainer - Container element for video
     * @returns {Array} - Array of position data for face indicators
     */
    static calculateFaceIndicators(detectionResult, videoContainer) {
        // Skip if detection unsuccessful or no container
        if (!detectionResult.success || !videoContainer) return [];
        
        const containerWidth = videoContainer.offsetWidth;
        const containerHeight = videoContainer.offsetHeight;
        
        // Map face locations to indicator positions
        return detectionResult.face_locations.map(location => {
            // Format from API is [top, right, bottom, left] as ratios of image dimensions
            const [top, right, bottom, left] = location;
            
            // Calculate pixel positions
            return {
                top: top * containerHeight,
                right: right * containerWidth,
                bottom: bottom * containerHeight,
                left: left * containerWidth,
                width: (right - left) * containerWidth,
                height: (bottom - top) * containerHeight
            };
        });
    }
    
    /**
     * Create face indicator overlays
     * @param {Array} indicators - Array of position data from calculateFaceIndicators
     * @param {HTMLElement} overlayContainer - Container to add indicators to
     */
    static createFaceOverlays(indicators, overlayContainer) {
        // Clear existing overlays
        overlayContainer.innerHTML = '';
        
        // Create new overlays
        indicators.forEach((pos, index) => {
            const indicator = document.createElement('div');
            indicator.className = 'face-indicator';
            
            // Position the indicator
            indicator.style.top = `${pos.top}px`;
            indicator.style.left = `${pos.left}px`;
            indicator.style.width = `${pos.width}px`;
            indicator.style.height = `${pos.height}px`;
            
            // Add data attributes for debugging
            indicator.dataset.faceIndex = index;
            
            // Add to container
            overlayContainer.appendChild(indicator);
        });
    }
    
    /**
     * Mark attendance for recognized students
     * @param {Array} studentIds - Array of student IDs to mark attendance for
     * @returns {Promise} - Attendance marking result
     */
    static async markAttendance(studentIds) {
        try {
            const response = await fetch('/mark_bulk_attendance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    student_ids: studentIds,
                    method: 'auto_recognition'
                })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error marking attendance:', error);
            return { success: false, error: error.message };
        }
    }
    
    /**
     * Verify if a face recognition result is reliable
     * @param {Object} result - Recognition result from API
     * @param {number} minConfidence - Minimum confidence threshold (0-100)
     * @returns {boolean} - Whether result is reliable
     */
    static isReliableRecognition(result, minConfidence = 70) {
        // Check if result is successful and has data
        if (!result.success || !result.data || !result.data.length) {
            return false;
        }
        
        // Check if any face exceeds the confidence threshold
        return result.data.some(face => face.confidence >= minConfidence);
    }
    
    /**
     * Extract reliable student IDs from recognition result
     * @param {Object} result - Recognition result from API
     * @param {number} minConfidence - Minimum confidence threshold (0-100)
     * @returns {Array} - Array of student IDs with confidence above threshold
     */
    static getReliableStudentIds(result, minConfidence = 80) {
        // Check if result is successful and has data
        if (!result.success || !result.data || !result.data.length) {
            return [];
        }
        
        // Filter for faces above confidence threshold
        return result.data
            .filter(face => face.confidence >= minConfidence)
            .map(face => face.student_id);
    }
}

// Export class for use in other files
window.FaceRecognitionMethods = FaceRecognitionMethods;