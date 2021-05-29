from flask import Flask, render_template, request, redirect, jsonify, redirect, url_for, flash
import business as business
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import requests
from torchvision import models
import torchvision.transforms as T
import numpy as np
import os

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
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
            img_url = filename
            business.add_background(img_url)
            return render_template('index.html')
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Image Upload</title>
    </head>
    <body>
        <h1>Image Upload</h1>
        <form method="POST" action="" enctype="multipart/form-data">
        <p><input type="file" name="file"></p>
        <p><input type="submit" value="Submit"></p>
        </form>
    </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")