import os 
import sys
import pandas as pd 
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import datetime
from keras import layers, models
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_injection import DataInjection
from src.components.model_trainer import ModelTrainer
from src.components.window_generator import WindowGenerator

@dataclass
class PredictPiplineConfig:
    open_model_path=os.path.join('modelhouse', 'open_model.h5')
    high_model_path=os.path.join('modelhouse', 'high_model.h5')
    low_model_path =os.path.join('modelhouse', 'low_model.h5')
    close_model_path=os.path.join('modelhouse', 'close_model.h5')
    latest_prediction_table_path=os.path.join('predictions', 'latest_prediction.csv')
    latest_prediction_graph_path= os.path.join('predictions', 'latest_prediction.png')

class PredictPipeline():
    def __init__(self):
        self.paths=PredictPiplineConfig()
        self.today = datetime.datetime.today()
        self.pred_step=pd.read_csv('dataset/pred_step.csv')
        self.model_dic={'Close':models.load_model(self.paths.open_model_path),
                        'High':models.load_model(self.paths.high_model_path),
                        'Low':models.load_model(self.paths.low_model_path),
                        'Open':models.load_model(self.paths.close_model_path)}

    def update_model(self):
        try:
            if self.today.weekday() < 5:
                return "SORRY MARKET IS IN SECTION AT THE MOMENT MODEL CAN NOT BE UPDATED DUE TO FLUCTATION IN MARKET PRICE KINDLY CHECK BACK WHEN THE MARKET IS CLOSED"
            else:               
                obj=DataInjection()
                obj.collect_train_data()
                obj.last_week_actual_price()
                win_obj=ModelTrainer()
                model_dic=win_obj.initiate_window_obj()
                win_obj.initiate_trainnig(model_dic)
                logging.info('model updated succefully')
                return "MODEL UPDATED SUCCEFULLY"
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_price(self):
        try:
            if self.today.weekday() < 5:
                    return pd.read_csv(self.paths.latest_prediction_table_path, index_col='Date')
            else:
                Date=[]
                latest_predictions={}
                now=datetime.datetime.now() 
                next_monday = now  + datetime.timedelta(days=(7 - now.weekday()))
                dates = [next_monday + datetime.timedelta(days=i) for i in range(5)]
                for date in dates:
                      Date.append(date.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + 'Z')
                Date=pd.to_datetime(Date)
                for column, model_ in self.model_dic.items():                  
                    predictions=model_.predict(np.expand_dims(self.pred_step, axis=0))
                    predictions=np.ravel(predictions)
                    predictions=(predictions*0.01)+1.1
                    latest_predictions[column]=predictions
                latest_predictions['Date']=Date
                latest_predictions['Volume']=[0.0,0.0,0.0,0.0,0.0]
                latest_predictions['Dividends']=[0.0,0.0,0.0,0.0,0.0]
                latest_predictions['Stock Splits']=[0.0,0.0,0.0,0.0,0.0]
                forcast=pd.DataFrame(latest_predictions)
                forcast.set_index('Date', inplace=True)
                forcast.to_csv(self.paths.latest_prediction_table_path, header=True, index= True)
                mpf.plot(forcast, type='candle', style='charles', volume=True, savefig=self.paths.latest_prediction_graph_path)
                return forcast
                
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=='__main__':
    obj=PredictPipeline()
    obj.predict_price()
    
                



    
