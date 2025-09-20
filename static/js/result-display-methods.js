/**
 * Result Display Methods
 * This file contains methods for displaying face recognition results
 */

class ResultDisplayMethods {
    /**
     * Displays the recognized faces in real-time on the UI
     * @param {Array} recognizedFaces - Array of recognized face objects 
     * @param {HTMLElement} container - The container element to display results in
     * @param {boolean} resultDisplayed - Whether results are currently being displayed
     * @returns {boolean} - Updated resultDisplayed status
     */
    static displayRealtimeResults(recognizedFaces, container, resultDisplayed = false) {
        if (!container) {
            container = document.getElementById('realTimeResults');
        }
        
        if (!container) return resultDisplayed;
        
        // Clear previous results if this is a new recognition cycle
        if (!resultDisplayed) {
            container.innerHTML = '';
            resultDisplayed = true;
        }
        
        // Process each recognized face
        recognizedFaces.forEach(face => {
            // Generate a unique ID for this result based on student ID and timestamp
            const resultId = `result-${face.student_id}-${Date.now()}`;
            
            // Check if we already displayed this person recently
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
                    ResultDisplayMethods.updateConfidenceClass(existingResult, face.confidence);
                }
                
                // Update timestamp
                const timestamp = existingResult.querySelector('.timestamp');
                if (timestamp) {
                    timestamp.textContent = `Updated: ${ResultDisplayMethods.getCurrentTime()}`;
                }
            } else {
                // Create new result card
                const resultHTML = ResultDisplayMethods.createResultCardHTML(face);
                
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
                ResultDisplayMethods.updateConfidenceClass(resultCard, face.confidence);
                
                // Prepend to show newest results at the top
                container.prepend(resultContainer);
                
                // Apply entrance animation
                setTimeout(() => {
                    resultContainer.style.opacity = '1';
                    resultContainer.style.transform = 'translateY(0)';
                }, 10);
                
                // Mark attendance automatically if callback is provided
                if (typeof window.markAttendanceCallback === 'function') {
                    window.markAttendanceCallback(face);
                }
            }
        });
        
        return resultDisplayed;
    }
    
    /**
     * Creates HTML for a result card
     * @param {Object} face - Face recognition data
     * @returns {string} - HTML string for the result card
     */
    static createResultCardHTML(face) {
        // Format confidence for display
        const confidence = Math.round(face.confidence);
        
        // Get current time
        const currentTime = ResultDisplayMethods.getCurrentTime();
        
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
    
    /**
     * Displays unknown faces in the UI
     * @param {number} faceCount - Number of unknown faces detected
     * @param {HTMLElement} container - The container element to display results in
     * @returns {boolean} - Whether results are now displayed
     */
    static displayUnknownFaces(faceCount, container = null) {
        if (!container) {
            container = document.getElementById('realTimeResults');
        }
        
        if (!container) return false;
        
        // Don't clear container if we have recognized results
        if (!container.querySelector('.realtime-result-card')) {
            // Create message for unknown face(s)
            const unknownHTML = `
                <div class="unknown-face">
                    <i class="fas fa-question-circle"></i>
                    <h4>${faceCount > 1 ? `${faceCount} Unknown Faces` : 'Unknown Face'}</h4>
                    <p>No match found in the database</p>
                </div>
            `;
            
            container.innerHTML = unknownHTML;
            return true;
        }
        
        return false;
    }
    
    /**
     * Updates confidence class on an element
     * @param {HTMLElement} element - Element to update class on 
     * @param {number} confidence - Confidence score (0-100)
     */
    static updateConfidenceClass(element, confidence) {
        // Remove all confidence classes first
        element.classList.remove('low-confidence', 'medium-confidence', 'high-confidence');
        
        // Add appropriate class based on confidence
        if (confidence >= 85) {
            element.classList.add('high-confidence');
        } else if (confidence >= 70) {
            element.classList.add('medium-confidence');
        } else {
            element.classList.add('low-confidence');
        }
    }
    
    /**
     * Gets formatted current time string
     * @returns {string} - Formatted time (HH:MM:SS AM/PM)
     */
    static getCurrentTime() {
        const now = new Date();
        let hours = now.getHours();
        let minutes = now.getMinutes();
        let seconds = now.getSeconds();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        
        hours = hours % 12;
        hours = hours ? hours : 12;
        minutes = minutes < 10 ? '0' + minutes : minutes;
        seconds = seconds < 10 ? '0' + seconds : seconds;
        
        return `${hours}:${minutes}:${seconds} ${ampm}`;
    }
    
    /**
     * Clears recognition results from container
     * @param {HTMLElement} container - Results container
     * @returns {boolean} - Whether results were cleared
     */
    static clearRecognitionResults(container = null) {
        if (!container) {
            container = document.getElementById('realTimeResults');
        }
        
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 20px;">Start recognition to see results</p>';
            return true;
        }
        
        return false;
    }
    
    /**
     * Marks attendance for multiple students
     * @param {Array} studentIds - Array of student IDs
     * @returns {Promise} - Attendance marking result
     */
    static markAttendanceForStudents(studentIds) {
        // Skip if empty
        if (!studentIds || !studentIds.length) return Promise.resolve({success: false, message: 'No student IDs provided'});
        
        console.log('Automatically marking attendance for:', studentIds);
        
        // Call API to mark attendance
        return fetch('/mark_bulk_attendance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                student_ids: studentIds,
                method: 'auto_recognition'
            })
        })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                console.log(`Attendance marked for ${result.marked_count} students`);
                // Show a subtle notification
                ResultDisplayMethods.showAttendanceMarkedNotification(result.marked_count);
            }
            return result;
        })
        .catch(error => {
            console.error('Error marking attendance:', error);
            return {success: false, error: error.message};
        });
    }
    
    /**
     * Shows attendance marked notification
     * @param {number} count - Number of students marked 
     */
    static showAttendanceMarkedNotification(count) {
        const container = document.createElement('div');
        container.className = 'attendance-marked-notification';
        container.innerHTML = `
            <div class="notification-icon"><i class="fas fa-check-circle"></i></div>
            <div class="notification-text">Attendance marked for ${count} student${count !== 1 ? 's' : ''}</div>
        `;
        
        document.body.appendChild(container);
        
        setTimeout(() => {
            container.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            container.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(container);
            }, 300);
        }, 3000);
    }
}

// Export class for use in other files
window.ResultDisplayMethods = ResultDisplayMethods;