document.getElementById('uploadForm').onsubmit = function(event) {
    event.preventDefault(); // Prevent the default form submission

    const formData = new FormData(this);
    const key = document.getElementById('keyInput').value;

    // Validate the key
    if (!key) {
        alert('Please enter a key.');
        return;
    }

    // Append the key to the form data
    formData.append('key', key);

    // Send the form data to the server
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Display the uploaded image and prediction result
            document.getElementById('resultImage').src = data.imageUrl;
            document.getElementById('predictionResult').innerText = data.prediction;
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while uploading the image.');
    });
};