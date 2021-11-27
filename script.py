#importing libraries
import os
import numpy as np
import math
import pandas as pd
import pickle
import flask
from pyspark import SparkContext
from flask import Flask, render_template, request
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors

template_dir = os.path.abspath('./templates')
static_dir = os.path.abspath('./templates/static')

app=Flask(__name__, template_folder=template_dir, static_folder=static_dir)



sc = SparkContext(appName="Research_project")



model = RandomForestClassificationModel.load('researchModels_RF')


#to tell flask what url shoud trigger the function index()
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        predict_list = list(map(int, predict_list))

        predicted = model.predict(Vectors.dense(predict_list))

        predictProb = model.predictProbability(Vectors.dense(predict_list))

        probability1 =  predictProb[1]*100

        probability0 =  predictProb[0]*100

        prob1 = f"{probability1:.2f}%"
        prob0 = f"{probability0:.2f}%"

        if int(predicted) == 1:
            prediction = 1
            comment = "The user is transacting"
        elif int(predicted) == 0:
            prediction = 0
            comment = "The user is not transacting"

    return render_template("index.html", prediction=prediction, comment=comment, prob0=prob0, prob1=prob1)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')