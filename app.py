'''
Created on 

@author: Bhavana, Santhosh

source:
    https://www.tutorialspoint.com/how-to-crop-and-save-the-detected-faces-in-opencv-python
    https://wpreset.com/force-reload-cached-css/

'''


import os
from os import listdir
from flask import Flask, flash, redirect, render_template, request, session
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import cv2
import math
import random
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

IMAGE_FOLDER    = os.getenv("IMAGE_FOLDER")
CROP_FOLDER     = "static/images/circle"
DETECTION_FILE  = "static/haarcascades/haarcascade_frontalface_default.xml"
CLOCK_PICTURE   = "static/images/clock.png"

def distance(x0, y0, x1, y1):
    p       = [x0, y0]
    q       = [x1, y1]
    print(math.dist(p, q))
    return int(math.dist(p, q))

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        name = request.form.get("name") 

        folder_dir = IMAGE_FOLDER + "/" + name
        count = 1
        co_ordinates = [(503, 31), (738, 89), (918, 266), (978, 500), (918, 735), (730, 870), (503, 930), (278, 890), (107, 735), (39, 495), (107, 280), (278, 90)]
        
        files = os.listdir(folder_dir)
        random.shuffle(files)
        print(files)
        list_of_images = []

        for image in files:
            if(count>12):
                break
            
            # read the input image
            img             = cv2.imread(f"{folder_dir}/{image}")

            # convert to grayscale of each frames
            gray            = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # read the haarcascade to detect the faces in an image
            face_cascade    = cv2.CascadeClassifier(DETECTION_FILE)

            # detects faces in the input image
            faces           = face_cascade.detectMultiScale(gray, 1.3, 4)
            # print('Number of detected faces:', len(faces))

            if(len(faces) == 1):
                list_of_images.append(image)
                for i, (x, y, w, h) in enumerate(faces):
                    # To draw a rectangle in a face
                    radius      = int((max(w, h)/2) +(max(w, h)/2))
                    centre_x    = int(x + w/2)
                    centre_y    = int(y + h/2)
                    #print(radius, (centre_x, centre_y), x, y, w, h)
                    # cv2.circle(img, (centre_x, centre_y), radius, (0,255,0), thickness=1)

                    # Open the input image as numpy array, convert to RGB
                    img         = Image.open(f"{folder_dir}/{image}").convert("RGB")
                    npImage     = np.array(img)
                    h, w        = img.size
                    #print("....?, ?", h, w)

                    # Create same size alpha layer with circle
                    alpha       = Image.new('L', img.size, 0)
                    draw        = ImageDraw.Draw(alpha)

                    left        = distance(centre_x, centre_y, 0, centre_y) - radius
                    top         = distance(centre_x, centre_y, centre_x, 0) - radius
                    right       = centre_x+radius
                    bottom      = centre_y+radius
                    #print(".....?, ?", left, top, bottom, right)
                    draw.pieslice((left, top, right, bottom), 0, 360, fill=255)
                    # Convert alpha Image to numpy array
                    npAlpha     = np.array(alpha)

                    # Add alpha layer to RGB
                    npImage     = np.dstack((npImage, npAlpha))

                    # Save with alpha
                    final       = Image.fromarray(npImage).crop(((left, top, right, bottom)))
                    h, w        = final.size
                    #print("................................",h,w)
                    final       = final.resize((208, 208))
                    final.save(f"{CROP_FOLDER}/face_{count}.png", quality=88)

                    print(f"{CROP_FOLDER}/face{count}.png")
                    # cv2.imwrite("static/images/circle_jenna/face.jpg", img)

                    img         = Image.open(f'{CROP_FOLDER}/face_{count}.png')
                    centre      = co_ordinates[count-1]
                    #print(centre)
                    mask_im     = Image.new("L", img.size, 0)
                    draw        = ImageDraw.Draw(mask_im)
                    draw.ellipse((1, 1, 208, 208), fill=255)
                    clock       = Image.open(CLOCK_PICTURE)
                    back_im     = clock.copy()
                    back_im.paste(img, centre, mask_im)
                    back_im.save(CLOCK_PICTURE)
                count += 1
            else:
                print("More than one")
                # print(f"{folder_dir}/{img}")

        # print(name)
        print(list_of_images)
        return render_template('index.html')
    else:
        names = [name for name in os.listdir("static/images/pics") if len(str(name).split("."))==1]
        return render_template('home.html', names=names)

@app.route('/index',methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pass
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5021)
