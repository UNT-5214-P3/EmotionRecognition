#!/usr/bin/env python
import os
import pickle
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify, Markup, send_file
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

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
    return render_template('result.html', url = filename)


# when running app locally
if __name__=='__main__':
      app.run(debug=True)