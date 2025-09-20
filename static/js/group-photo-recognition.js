/**
 * Group Photo Recognition JavaScript
 * Handles uploading and processing group photos for face recognition
 */

class GroupPhotoRecognition {
    constructor() {
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupDropZone();
    }
    
    setupEventListeners() {
        // Single photo upload
        const singlePhotoInput = document.getElementById('singlePhotoInput');
        const singleUploadBtn = document.getElementById('uploadSinglePhoto');
        
        if (singlePhotoInput && singleUploadBtn) {
            singleUploadBtn.addEventListener('click', () => singlePhotoInput.click());
            singlePhotoInput.addEventListener('change', (e) => this.handleSinglePhotoUpload(e));
        }
        
        // Group photo upload
        const groupPhotoInput = document.getElementById('groupPhotoInput');
        const groupUploadBtn = document.getElementById('uploadGroupPhoto');
        
        if (groupPhotoInput && groupUploadBtn) {
            groupUploadBtn.addEventListener('click', () => groupPhotoInput.click());
            groupPhotoInput.addEventListener('change', (e) => this.handleGroupPhotoUpload(e));
        }
        
        // Clear results button
        const clearBtn = document.getElementById('clearResults');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearResults());
        }
    }
    
    setupDropZone() {
        const dropZones = document.querySelectorAll('.upload-area');
        
        dropZones.forEach(dropZone => {
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });
            
            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('drag-over');
            });
            
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const file = files[0];
                    if (file.type.startsWith('image/')) {
                        const isGroup = dropZone.id === 'groupPhotoDropZone';
                        this.processUploadedFile(file, isGroup);
                    } else {
                        this.showAlert('Please upload an image file', 'error');
                    }
                }
            });
        });
    }
    
    handleSinglePhotoUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.processUploadedFile(file, false);
        }
    }
    
    handleGroupPhotoUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.processUploadedFile(file, true);
        }
    }
    
    async processUploadedFile(file, isGroup) {
        try {
            // Show loading state
            this.showLoadingState(isGroup);
            
            // Create FormData
            const formData = new FormData();
            formData.append('image', file);
            formData.append('is_group', isGroup.toString());
            
            // Upload and process
            const response = await fetch('/recognize_upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();

            // Enhanced debugging
            console.log('API Response:', result);
            console.log('Response success:', result.success);
            console.log('Response message:', result.message);

            if (result.success) {
                console.log('Success path - displaying results');
                if (isGroup) {
                    this.displayGroupResults(result.results, result.image_path);
                } else {
                    this.displaySingleResult(result.match, result.image_path);
                }
            } else {
                console.log('Failed path - showing no match result');
                // Show the error message but also display what we received for debugging
                console.log('Recognition failed. Full response:', result);

                // Always show no-match result for failed recognition
                if (isGroup) {
                    // Show empty group results
                    this.displayGroupResults([], result.image_path || 'test-image.jpg');
                } else {
                    // Show no match single result
                    this.displayNoMatchResult();
                }

                // Also show alert
                this.showAlert(result.message || 'Recognition failed', 'warning');
            }
            
        } catch (error) {
            console.error('Error processing upload:', error);
            this.showAlert('Error processing image', 'error');
        } finally {
            this.hideLoadingState();
        }
    }
    
    showLoadingState(isGroup) {
        const resultsContainer = document.getElementById('recognitionResults');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <p>Processing ${isGroup ? 'group photo' : 'image'}...</p>
                    <p class="text-muted">This may take a moment</p>
                </div>
            `;
        }
    }
    
    hideLoadingState() {
        // Loading state will be replaced by results
    }
    
    displayNoMatchResult() {
        console.log('displayNoMatchResult called');
        const resultsContainer = document.getElementById('recognitionResults');
        console.log('Results container found:', !!resultsContainer);
        if (!resultsContainer) return;

        console.log('Setting no-match HTML content');
        resultsContainer.innerHTML = `
            <div class="recognition-result single-result no-match">
                <h3><i class="fas fa-user-times"></i> Recognition Result</h3>

                <div class="no-match-message">
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle"></i>
                        <strong>No Match Found</strong>
                        <p>The uploaded image could not be matched to any student in the database.</p>
                        <p class="text-muted">This could be because:</p>
                        <ul class="text-muted">
                            <li>The person is not registered in the system</li>
                            <li>The image quality is too low</li>
                            <li>The face is not clearly visible</li>
                        </ul>
                    </div>
                </div>

                <div class="actions mt-3">
                    <button class="btn btn-primary" onclick="document.getElementById('singlePhotoInput').click()">
                        <i class="fas fa-upload"></i> Try Another Image
                    </button>
                    <button class="btn btn-secondary" onclick="window.groupPhotoRecognition.clearResults()">
                        <i class="fas fa-times"></i> Clear Results
                    </button>
                </div>
            </div>
        `;
    }

    displaySingleResult(match, imagePath) {
        console.log('displaySingleResult called with:', match, imagePath);
        const resultsContainer = document.getElementById('recognitionResults');
        console.log('Results container found:', !!resultsContainer);
        if (!resultsContainer || !match) {
            console.log('No container or no match, calling displayNoMatchResult');
            this.displayNoMatchResult();
            return;
        }

        const confidence = match.confidence || Math.round(match.similarity * 100);
        const confidenceClass = confidence >= 80 ? 'high' : confidence >= 60 ? 'medium' : 'low';
        
        resultsContainer.innerHTML = `
            <div class="recognition-result single-result">
                <h3><i class="fas fa-user-check"></i> Recognition Result</h3>
                
                <div class="result-grid">
                    <div class="uploaded-image">
                        <img src="/static/uploads/${imagePath}" alt="Uploaded Image" class="img-fluid">
                        <p class="text-center mt-2">Uploaded Image</p>
                    </div>
                    
                    <div class="match-details">
                        <div class="student-card ${confidenceClass}-confidence">
                            <div class="student-info">
                                <h4>${match.name}</h4>
                                <p class="student-id">ID: ${match.student_id}</p>
                                <p class="student-email">${match.email || 'N/A'}</p>
                                <p class="student-phone">${match.phone || 'N/A'}</p>
                            </div>
                            
                            <div class="confidence-display">
                                <div class="confidence-bar">
                                    <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
                                </div>
                                <p class="confidence-text">${confidence}% Match</p>
                            </div>
                        </div>
                        
                        <button class="btn btn-success mt-3" onclick="markAttendance('${match.student_id}', '${match.name}')">
                            <i class="fas fa-check"></i> Mark Attendance
                        </button>
                    </div>
                </div>
            </div>
        `;
    }
    
    displayGroupResults(results, imagePath) {
        const resultsContainer = document.getElementById('recognitionResults');
        if (!resultsContainer) return;
        
        const recognizedCount = results.filter(r => r.match).length;
        const unrecognizedCount = results.length - recognizedCount;
        
        let html = `
            <div class="recognition-result group-result">
                <h3><i class="fas fa-users"></i> Group Photo Recognition Results</h3>
                
                <div class="results-summary">
                    <div class="summary-stats">
                        <div class="stat">
                            <span class="stat-number">${results.length}</span>
                            <span class="stat-label">Faces Detected</span>
                        </div>
                        <div class="stat">
                            <span class="stat-number">${recognizedCount}</span>
                            <span class="stat-label">Recognized</span>
                        </div>
                        <div class="stat">
                            <span class="stat-number">${unrecognizedCount}</span>
                            <span class="stat-label">Unknown</span>
                        </div>
                    </div>
                </div>
                
                <div class="group-image-container">
                    <img src="/static/uploads/${imagePath}" alt="Group Photo" class="img-fluid">
                </div>
                
                <div class="recognized-faces">
                    <h4>Recognized Students</h4>
                    <div class="faces-grid">
        `;
        
        // Add recognized faces
        results.forEach((result) => {
            if (result.match) {
                const confidence = result.match.confidence || Math.round(result.match.similarity * 100);
                const confidenceClass = confidence >= 80 ? 'high' : confidence >= 60 ? 'medium' : 'low';
                
                html += `
                    <div class="face-result ${confidenceClass}-confidence">
                        <div class="face-info">
                            <h5>${result.match.name}</h5>
                            <p class="student-id">ID: ${result.match.student_id}</p>
                            <div class="confidence-bar">
                                <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
                            </div>
                            <p class="confidence-text">${confidence}% Match</p>
                        </div>
                        
                        <button class="btn btn-sm btn-success" onclick="markAttendance('${result.match.student_id}', '${result.match.name}')">
                            <i class="fas fa-check"></i> Mark Attendance
                        </button>
                    </div>
                `;
            }
        });
        
        html += `
                    </div>
                </div>
                
                <div class="bulk-actions mt-4">
                    <button class="btn btn-primary" onclick="markAllAttendance()">
                        <i class="fas fa-check-double"></i> Mark All Attendance
                    </button>
                </div>
            </div>
        `;
        
        resultsContainer.innerHTML = html;
        
        // Store results for bulk actions
        window.groupRecognitionResults = results.filter(r => r.match);
    }
    
    clearResults() {
        const resultsContainer = document.getElementById('recognitionResults');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-image fa-3x mb-3"></i>
                    <p>Upload an image to see recognition results</p>
                </div>
            `;
        }
    }
    
    showAlert(message, type = 'info') {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at top of container
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(alert, container.firstChild);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.remove();
                }
            }, 5000);
        }
    }
}

// Attendance marking functions
async function markAttendance(studentId, studentName) {
    try {
        const response = await fetch('/mark_attendance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                student_id: studentId,
                name: studentName
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update UI to show attendance marked
            const button = event.target.closest('button');
            if (button) {
                button.innerHTML = '<i class="fas fa-check"></i> Attendance Marked';
                button.className = 'btn btn-sm btn-secondary';
                button.disabled = true;
            }
            
            // Show success message
            window.groupPhotoRecognition.showAlert(`Attendance marked for ${studentName}`, 'success');
        } else {
            window.groupPhotoRecognition.showAlert(result.message || 'Failed to mark attendance', 'error');
        }
    } catch (error) {
        console.error('Error marking attendance:', error);
        window.groupPhotoRecognition.showAlert('Error marking attendance', 'error');
    }
}

async function markAllAttendance() {
    if (!window.groupRecognitionResults || window.groupRecognitionResults.length === 0) {
        return;
    }
    
    try {
        let successCount = 0;
        let totalCount = window.groupRecognitionResults.length;
        
        for (const result of window.groupRecognitionResults) {
            const response = await fetch('/mark_attendance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    student_id: result.student_id,
                    name: result.name
                })
            });
            
            const attendanceResult = await response.json();
            if (attendanceResult.success) {
                successCount++;
            }
        }
        
        // Update UI
        const buttons = document.querySelectorAll('.face-result button');
        buttons.forEach(button => {
            button.innerHTML = '<i class="fas fa-check"></i> Attendance Marked';
            button.className = 'btn btn-sm btn-secondary';
            button.disabled = true;
        });
        
        // Update bulk action button
        const bulkButton = document.querySelector('.bulk-actions button');
        if (bulkButton) {
            bulkButton.innerHTML = '<i class="fas fa-check-double"></i> All Attendance Marked';
            bulkButton.className = 'btn btn-secondary';
            bulkButton.disabled = true;
        }
        
        window.groupPhotoRecognition.showAlert(`Attendance marked for ${successCount} out of ${totalCount} students`, 'success');

    } catch (error) {
        console.error('Error marking bulk attendance:', error);
        window.groupPhotoRecognition.showAlert('Error marking bulk attendance', 'error');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.groupPhotoRecognition = new GroupPhotoRecognition();
});