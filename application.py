from flask import Flask, request, render_template, send_file, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from src.pipeline.predict_pipeline import PredictPipeline

application=Flask(__name__)
app=application
@app.route('/') #route for home page
def index():
    return render_template('test.html')

@app.route('/predict', methods=['GET'])
def predict():
    obj=PredictPipeline()
    prediction = obj.predict_price()
    return render_template('predict.html', table=prediction,  plot=url_for('predict', filename='latest_prediction.png', mimetype='image/png'), prediction = prediction)
    #C:\Users\owner\MODEL_eurusd\finance\lib\site-packages\mplfinance\plotting.py
    #C:\Users\owner\MODEL_eurusd\predictions\latest_prediction.png
@app.route('/update', methods=['GET'])
def update():
    obj=PredictPipeline()
    notice=obj.update_model()
    return render_template('update.html', result=notice)
@app.route('/compare', methods=['GET'])
def compare():
    prediction= pd.read_csv('prediction\latest_prediction.csv', index_col='Date')
    actual=pd.read_csv('dataset\last_week_actual_price.csv', index_col='Date')
    mpf.plot(actual, type='candle', style='charles', volume=True, savefig='prediction/latest_actual.png')
    return render_template('test.html', table1=prediction.to_html(), table2=actual.to_html(), plot1='prediction/latetst_prediction.png', plot2='prediction/latest_actual.png')



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)