function captureFrame() {
    fetch('/capture', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerText = 'Error: ' + data.error;
        } else {
            document.getElementById('captured-image').src = 'data:image/jpeg;base64,' + data.image;
            document.getElementById('result').innerText = '';
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function classifyImage() {
    const imgElement = document.getElementById('captured-image');
    const imageBase64 = imgElement.src.split(',')[1]; // Get the base64 part

    fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageBase64 })
    })
    .then(response => response.json())
    .then(data => {
        if (data.gender) {
            document.getElementById('result').innerText = `Predicted Gender: ${data.gender}, Probability: ${data.probability.toFixed(2)}`;
        } else {
            document.getElementById('result').innerText = 'Error: Unable to classify';
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

