import os
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from mtcnn import MTCNN



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the saved model
model = tf.keras.models.load_model('model/gender_classifier-with50-epoch.h5', compile=False)
# Compile the model manually
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Function to preprocess the frame
# def preprocess_frame(frame):
#     img_array = cv2.resize(frame, (64, 64))
#     img_array = image.img_to_array(img_array)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Rescale the image to match training conditions
#     return img_array

def preprocess_frame(frame):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    if faces:
        bounding_box = faces[0]['box']  # Assuming only one face
        x, y, w, h = bounding_box
        face_img = frame[y:y+h, x:x+w]
        img_array = cv2.resize(face_img, (64, 64))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale the image to match training conditions
        return img_array
    else:
        # If no face is detected, return None or handle the case accordingly
        return None


# def predict_gender(frame, male_threshold=0.6, female_threshold=0.4):
#     preprocessed_frame = preprocess_frame(frame)
#     prediction = model.predict(preprocessed_frame)
#     probability_male = prediction[0][0]
#     probability_female = 1 - probability_male
    
#     if probability_male >= male_threshold:
#         gender = 'Male'
#         probability = probability_male
#     elif probability_female >= female_threshold:
#         gender = 'Female'
#         probability = probability_female
#     else:
#         gender = 'Unclear'
#         probability = max(probability_male, probability_female)
    
#     return gender, probability

def predict_gender(frame, male_threshold=0.6, female_threshold=0.4):
    preprocessed_frame = preprocess_frame(frame)
    if preprocessed_frame is not None:
        prediction = model.predict(preprocessed_frame)
        probability_male = prediction[0][0]
        probability_female = 1 - probability_male

        if probability_male >= male_threshold:
            gender = 'Male'
            probability = probability_male
        elif probability_female >= female_threshold:
            gender = 'Female'
            probability = probability_female
        else:
            gender = 'Unclear'
            probability = max(probability_male, probability_female)

        return gender, probability
    else:
        return 'No face detected', 0.0




# Function to generate frames from the webcam
def gen_frames():
    global camera_frame
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            camera_frame = frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global camera_frame
    if camera_frame is not None:
        ret, buffer = cv2.imencode('.jpg', camera_frame)
        frame = buffer.tobytes()
        frame_base64 = base64.b64encode(frame).decode('utf-8')
        return jsonify({'image': frame_base64})
    return jsonify({'error': 'No frame captured'})


@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    image_data = data['image']
    image_data = base64.b64decode(image_data)
    nparr = np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gender, probability = predict_gender(img)
    
    # Convert probability to Python float
    probability = float(probability)
    
    return jsonify({'gender': gender, 'probability': probability})



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            gender = predict_gender(cv2.imread(filepath))
            return render_template('upload.html', filename=filename, gender=gender)
    return render_template('upload.html', filename=None, gender=None)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    global camera_frame
    camera_frame = None
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))




