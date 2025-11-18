document.addEventListener('DOMContentLoaded', function() {
    const videoFeed = document.getElementById('videoFeed');
    const gestureText = document.getElementById('gestureText');
    const statusGesture = document.getElementById('statusGesture');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const cameraStatus = document.getElementById('cameraStatus');
    
    
    loadingSpinner.classList.add('active');

    videoFeed.onload = function() {
        console.log('Video feed loaded');
        loadingSpinner.classList.remove('active');
    };
    
    
    const updateGesture = async () => {
        try {
            const response = await fetch('/api/gesture');
            const data = await response.json();
            
            if (data.gesture !== 'No hands detected') {
                gestureText.textContent = data.gesture;
                statusGesture.textContent = data.gesture;
            } else {
                gestureText.textContent = 'Waiting...';
                statusGesture.textContent = 'None';
            }
        } catch (error) {
            console.error('Error fetching gesture:', error);
            cameraStatus.textContent = '✗ Error';
            cameraStatus.classList.remove('active');
        }
    };
    
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            console.log('Backend health:', data);
            updateGesture();
        })
        .catch(error => {
            console.error('Backend error:', error);
        });
    
    // Set interval for continuous updates
    setInterval(updateGesture, 500);
});

function handleVideoError() {
    console.error('Video feed failed to load');
    document.getElementById('loadingSpinner').textContent = 'Camera not available';
}
