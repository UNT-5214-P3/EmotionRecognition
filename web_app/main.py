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
from PIL import Image
import torch
import torchvision.transforms as transforms
import boto3


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp'}

# load image and transform to tensor (for cnn-pytorch)
loader = transforms.Compose([transforms.ToTensor()])

# load cnn-resnet (pytorch) model
net = pickle.load(open("P3ModelPytorch.pkl", "rb"))

# load keras model
json_file = open('keras_model.json', 'r')
keras_model_json = json_file.read()
json_file.close()
keras_model = model_from_json(keras_model_json)

# load weights into keras model
keras_model.load_weights("keras_model_weights.h5") 

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

    # build data frame that store result for image classification
    df_predictions = pd.DataFrame(columns=['Model', 'Predicted Emotion', 'Probability'])

    df_predictions = cnn_pytorch_predict(filename, df_predictions)
    df_predictions = cnn_keras_predict(filename, df_predictions)
    df_predictions = aws_rekognition_classify(filename, df_predictions)
    
    return render_template('result.html', url=filename, predictions=df_predictions)

def cnn_keras_predict(filename, df_predictions):
    #load image
    full_size_image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # convert image to gray scale, then resize
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (WIDTH, HEIGHT)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        
    # predict the emotion by cnn (keras) model
    keras_yhat= keras_model.predict(cropped_img)
    pred_index = int(np.argmax(keras_yhat))
    keras_predict = LABELS[pred_index]
    keras_predict_proba = keras_yhat[0][pred_index]

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'CNN-VGG16 (keras)', \
                                            'Predicted Emotion': keras_predict,\
                                            'Probability': keras_predict_proba}, ignore_index=True)
    return df_predictions

def cnn_pytorch_predict(filename, df_predictions):
    image = image_loader(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # predict the emotion by cnn-resnet (pytorch) model
    outputs = net(image)
    _, predicted = torch.max(outputs, 1)

    # get classification probability
    sm = torch.nn.Softmax(dim=1)
    prob = sm(outputs)

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'CNN-Resnet (pytorch)', \
                                            'Predicted Emotion': LABELS[predicted.item()],\
                                            'Probability': '{:f}'.format(max(list(max(prob.data))))}, ignore_index=True)
    return df_predictions

def aws_rekognition_classify(filename, df_predictions):

    #initialize rekogniton client
    rekognition = boto3.client('rekognition')

    #invoke aws rekognition api
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as image_data:
        response_content = image_data.read()
        rekognition_response = rekognition.detect_faces(Image={'Bytes':response_content}, Attributes=['ALL'])
    
    #remove confused class
    model_reponse_type_output = []
    for class_type in rekognition_response['FaceDetails'][0]['Emotions']:
        if class_type['Type'] != 'CONFUSED':
            model_reponse_type_output.append(class_type)

    #extract highest confidence emotion
    predicted_type = ''
    confidence_value = 0
    for emotion in model_reponse_type_output:
        if confidence_value <= emotion['Confidence']:
            confidence_value = emotion['Confidence']
            predicted_type = emotion['Type']

    #convert response data similar to other models
    if(predicted_type == 'CALM'):
        predicted_type = 'NEUTRAL'
    elif(predicted_type == 'DISGUSTED'):
        predicted_type = 'DISGUST'
    elif(predicted_type == 'SURPRISED'):
        predicted_type = 'SURPRISE'
        
    predicted_type = predicted_type.capitalize()

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Amazon Rekognition API', \
                                            'Predicted Emotion': predicted_type,\
                                            'Probability': confidence_value * 0.01}, ignore_index=True)
    return df_predictions


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU

# when running app locally
if __name__=='__main__':
      app.run(debug=False, threaded=False)