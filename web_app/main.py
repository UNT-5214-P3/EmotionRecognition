#!/usr/bin/env python
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify, Markup, send_file
from werkzeug.utils import secure_filename
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp'}

# load keras model
json_file = open('keras_model.json', 'r')
keras_model_json = json_file.read()
json_file.close()
keras_model = model_from_json(keras_model_json)

# load weights into keras model
keras_model.load_weights("keras_model_weights.h5")

# load file for multiple facial detection in image
FACE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')     

#setting image resizing parameters
WIDTH = HEIGHT = 48
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('display_result',
                                    filename=filename))
    return render_template('upload.html')

@app.route('/url', methods=['GET', 'POST'])
def check_url():
    return render_template('url.html')


@app.route('/api/images', methods=['GET', 'POST'])
def images():
     if request.method == 'GET':
        filename = request.args.get('filename')
        return send_file('uploads/'+filename, mimetype='image/gif') 

@app.route('/result', methods=['GET'])
def display_result():
    filename = request.args.get('filename')

    #load image
    full_size_image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # convert image to gray scale
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)                   

    # detect faces in image and classify emotion for each face
    index = 0
    df_predictions = pd.DataFrame(columns=['Model', 'Face', 'Predicted Emotion', 'Probability'])
    faces = FACE.detectMultiScale(gray, 1.3, 10)
    for (x, y, w, h) in faces:
        index += 1
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (WIDTH, HEIGHT)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

        # draw rectangle box around face in the image
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # add face number above the rectangle
        cv2.putText(full_size_image, 'F' + str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        #predicting the emotion by keras model
        keras_yhat= keras_model.predict(cropped_img)
        keras_predict = LABELS[int(np.argmax(keras_yhat))]
        keras_predict_proba = dict(zip(LABELS, list(keras_yhat[0])))

        # save prediction to data frame
        df_predictions = df_predictions.append({'Model': 'Model 1 (keras)', \
                                            'Face': index,\
                                            'Predicted Emotion': keras_predict,\
                                            'Probability': keras_predict_proba}, ignore_index=True)

    # save image with drawn-rectangle(s)
    filename1 = 'rec_' + filename
    _ = cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename1), full_size_image)
    
    return render_template('result.html', url=filename1, predictions=df_predictions)


# when running app locally
if __name__=='__main__':
      app.run(debug=False, threaded=False)