from flask import Flask,flash,redirect,url_for,render_template,request
import cv2
import urllib.request
import numpy as np
import face_recognition
from datetime import datetime
import os
import pandas as pd
from werkzeug.utils import secure_filename




app = Flask(__name__)

UPLOAD_FOLDER = 'attendance_images'

app.config['SECRET_KEY'] = 'dineshchakri'
app.config['UPLOAD_PATH'] = 'attendance_images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

df = pd.read_csv('attendance.csv')


@app.route('/show_data',  methods=("POST", "GET"))
def showData():
    # Convert pandas dataframe to html table flask
    df_html = df.to_html()
    return render_template('index.html', data=df_html)


@app.route('/', methods=["GET"])
def home():
     return render_template('index.html')


@app.route('/Upload',methods=["GET","POST"])
def upload_file():
    if request.method == 'POST' :
        f = request.files['file-name']
        f.save(os.path.join(app.config['UPLOAD_PATH'],f.filename))
        return render_template('index.html' , msg="file uploaded successfully")
    return render_template('index.html',msg="please choose a file")

@app.route('/detection')
def Attendance():

    path = 'attendance_images'
    images = []
    classNames = []
    x = " "
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    def findEncodings(image):
        encodeList = []
        for img in image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def mark_attendance(name):
        with open('attendance.csv','r+') as f:
            myDataList = f.readlines()
            namelist = []
            for line in myDataList:
                entry = line.split(',')
                namelist.append(entry[0])
            if name not in namelist:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

    encodeListKnown = findEncodings(images)
    print('encoding complete')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
            success, img = cap.read()
            if img is None:
                print('Wrong path:')
            else:
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    x = name

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
                    mark_attendance(name)
                cv2.imshow('webcam', img)
                cv2.waitKey(1)

            return render_template('index.html', check=x)

if __name__ == '__main__':
    app.run(debug=True)
