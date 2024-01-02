from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Load the trained character recognition model
character_model = tf.keras.models.load_model('your_model.h5')

# List of all digits and characters
digits = list(map(str, range(10))) + ["ba", "pa"]

def recognize_individual_character(char_img):
    # Preprocess the character image for model input
    char_img = cv2.resize(char_img, (128, 128))
    char_img = char_img.astype(np.float32) / 255.0
    char_img = np.expand_dims(char_img, axis=0)
    
    # Make a prediction using the character model
    char_prediction = character_model.predict(char_img)
    char_class = np.argmax(char_prediction)
    char_label = digits[char_class]
    
    return char_label


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            # Save the uploaded image in your project directory
            image_path = os.path.join('uploaded_images', uploaded_file.filename)
            uploaded_file.save(image_path)
            
            # Load the image using PIL
            img = Image.open(image_path)
            
            # Apply license plate segmentation to get individual character images
            sample_char_image = cv2.imread(image_path)  # Replace with a sample character image
            recognized_char = recognize_individual_character(sample_char_image)
            
            return render_template('index.html', prediction=recognized_char, uploaded=True, uploaded_image=image_path)
    
    return render_template('index.html', uploaded=False)



if __name__ == '__main__':
    app.run(debug=True)
