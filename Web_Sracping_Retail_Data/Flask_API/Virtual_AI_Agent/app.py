# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:50:05 2020

@author: SDOSI
"""
import os
from flask import Flask, request, redirect, url_for, send_from_directory,render_template
from werkzeug.utils import secure_filename

import cv2 
from tensorflow import keras
import numpy as np

MODEL_PATH = 'C:\\Users\\sdosi\\ascena_work\\deep_learning\\Web_Sracping_Retail_Data\\Model_Save\\retail_img_clf_cnn_1.h5'
UPLOAD_FOLDER = 'C:\\Users\\sdosi\\ascena_work\\deep_learning\\Web_Sracping_Retail_Data\\Flask_API\\Virtual_AI_Agent\\uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','PNG','JPG'])
cat_dic = {0:'Jeans',1:'Shoes',2:'Sunglasses',3:'Tops'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_predict_value(IMG_PATH):
    model = keras.models.load_model(MODEL_PATH)
    image = cv2.imread(IMG_PATH) 
    new_image = cv2.resize(image,(150,150)) 
    input_arr = keras.preprocessing.image.img_to_array(new_image)
    #input_arr = input_arr*1/255.0
    input_arr = np.array([input_arr])
    # Convert single image to a batch.
    predictions = model.predict_proba(input_arr)
    predict = cat_dic[np.argmax(predictions[0])]
    return predict

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        #print(file)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            predict = get_predict_value(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('#'*40)
            print('predict : ',predict)
            return redirect(url_for('uploaded_file', filename=filename,prediction=predict))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/show/<filename>/<prediction>')
def uploaded_file(filename,prediction):
    filename = 'http://127.0.0.1:5000/uploads/' + filename
    print(prediction)
    return render_template('template.html', filename=filename,prediction=prediction)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run()
