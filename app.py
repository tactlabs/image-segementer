from flask import Flask, render_template, request, redirect, jsonify
import business as business
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import requests
from torchvision import models
import torchvision.transforms as T
import numpy as np
import os
  
app = Flask(__name__)
PORT = 3000

app.config["UPLOAD_FOLDER"] = "uploads"
  
@app.route("/2")
def home_view():

    return render_template ('index.html')

@app.route('/create',methods=["GET", "POST"])
def details():
    if request.method == "POST":
       img_url = request.form.get("img_url")
       bg_url = request.form.get("bg_url")
       business.add_background(img_url,bg_url)
       return 'check console'
       
    return 'wait pannungo'




if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=PORT)