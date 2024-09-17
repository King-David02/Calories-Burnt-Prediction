import pandas as pd
import pickle
from flask import Flask
from flask import request
from flask import jsonify
from model import Scaling


with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)


app = Flask('Calories Burnt')
@app.route('/app', methods = ['POST'])


def predict():