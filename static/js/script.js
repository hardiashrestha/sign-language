document.addEventListener('DOMContentLoaded', function() {
    const gestureText = document.getElementById('gestureText');
    const statusGesture = document.getElementById('statusGesture');
    
    // Update gesture text every 500ms
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
        }
    };
    
    // Initial update
    updateGesture();
    
    // Set interval for continuous updates
    setInterval(updateGesture, 500);
});
