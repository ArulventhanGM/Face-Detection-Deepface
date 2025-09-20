/**
 * Face Recognition Integration Test
 * This file provides functions to test the integration between our face recognition components
 */

class FaceRecognitionTest {
    static testClassIntegration() {
        console.log('Testing integration between face recognition components...');
        
        // Test FaceRecognitionMethods class
        if (typeof FaceRecognitionMethods !== 'undefined') {
            console.log('✓ FaceRecognitionMethods class is available');
        } else {
            console.error('✗ FaceRecognitionMethods class is not available');
        }
        
        // Test ResultDisplayMethods class
        if (typeof ResultDisplayMethods !== 'undefined') {
            console.log('✓ ResultDisplayMethods class is available');
        } else {
            console.error('✗ ResultDisplayMethods class is not available');
        }
        
        // Test RealtimeFaceRecognition class
        if (typeof RealtimeFaceRecognition !== 'undefined') {
            console.log('✓ RealtimeFaceRecognition class is available');
        } else {
            console.error('✗ RealtimeFaceRecognition class is not available');
        }
    }
    
    static testUIComponents() {
        console.log('Testing UI components...');
        
        // Test camera preview
        const cameraPreview = document.getElementById('cameraPreview');
        if (cameraPreview) {
            console.log('✓ Camera preview element exists');
        } else {
            console.error('✗ Camera preview element not found');
        }
        
        // Test camera controls
        const startCamera = document.getElementById('startCamera');
        const stopCamera = document.getElementById('stopCamera');
        if (startCamera && stopCamera) {
            console.log('✓ Camera control buttons exist');
        } else {
            console.error('✗ Camera control buttons not found');
        }
        
        // Test recognition controls
        const startRecognition = document.getElementById('startRecognition');
        const stopRecognition = document.getElementById('stopRecognition');
        if (startRecognition && stopRecognition) {
            console.log('✓ Recognition control buttons exist');
        } else {
            console.error('✗ Recognition control buttons not found');
        }
        
        // Test results container
        const realTimeResults = document.getElementById('realTimeResults');
        if (realTimeResults) {
            console.log('✓ Real-time results container exists');
        } else {
            console.error('✗ Real-time results container not found');
        }
        
        // Test streaming status
        const streamingStatus = document.getElementById('streamingStatus');
        if (streamingStatus) {
            console.log('✓ Streaming status element exists');
        } else {
            console.error('✗ Streaming status element not found');
        }
    }
    
    static runAllTests() {
        console.log('Running all face recognition integration tests...');
        this.testClassIntegration();
        this.testUIComponents();
        console.log('Integration tests completed!');
    }
}

// Run tests when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Only run tests if we're on the recognition page
    if (document.getElementById('cameraPreview')) {
        setTimeout(() => {
            FaceRecognitionTest.runAllTests();
        }, 500); // Small delay to ensure all scripts are loaded
    }
});