// Upload form handling
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.querySelector('form');
    const progressBar = document.querySelector('#upload-progress');
    const processingStatus = document.querySelector('#processing-status');

    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const formData = new FormData(this);

            // Show progress bar
            progressBar.classList.remove('d-none');

            // Upload file
            fetch(uploadForm.action, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Show processing status
                    processingStatus.classList.remove('d-none');

                    // Poll for processing status
                    pollProcessingStatus(data.audio_id);
                } else {
                    alert(data.error || 'Upload failed');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Upload failed');
            })
            .finally(() => {
                progressBar.classList.add('d-none');
            });
        });
    }
});

// Poll for processing status
function pollProcessingStatus(audioId) {
    const statusText = document.querySelector('#status-text');
    const processingBar = document.querySelector('#processing-status .progress-bar');

    function checkStatus() {
        fetch(`/api/status/${audioId}`)
            .then(response => response.json())
            .then(data => {
                if (data.processed) {
                    // Redirect to results page
                    window.location.href = `/results/${data.result_id}`;
                } else {
                    // Update progress bar
                    processingBar.style.width = '50%';
                    statusText.textContent = 'Processing audio...';

                    // Continue polling
                    setTimeout(checkStatus, 2000);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                statusText.textContent = 'Error checking status';
            });
    }

    checkStatus();
}

// Initialize tooltips
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl)
});