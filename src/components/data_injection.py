import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from src.exception import CustomException 
from src.logger import logging 
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DataInjectionConfig:
    train_data_path:str = os.path.join('dataset', 'train_dataset.csv')
    last_week_actual_price: str= os.path.join('dataset', 'last_week_actual_price.csv')
    pictorial_view :str= os.path.join('dataset', 'last_week_actual_price.png')

class DataInjection:
    def __init__(self):
        self.data_injection= DataInjectionConfig()

    def collect_train_data(self):
        try:
            logging.info("collect the updated version of data and read to dataframe")
            eurusd = yf.Ticker("EURUSD=X")
            df = eurusd.history(period="max")
            logging.info("resetting index and droping un-used feature")
            df=df.reset_index()
            df=df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)
            logging.info('starts feature engineering on dataset')
            df['Date']=pd.to_datetime(df['Date'], utc=True)
            df['hour_sin'] = np.sin(2 * np.pi * df['Date'].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['Date'].dt.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofyear / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofyear / 365)
            df['year_sin'] = np.sin(2 * np.pi * df['Date'].dt.year / (10*365))
            df['year_cos'] = np.cos(2 * np.pi * df['Date'].dt.year / (10*365))
            df.pop('Date')
            logging.info("normalizing the data has commence")
            mean, std = df.mean(), df.std()
            df = (df - mean) / std
            logging.info("dataset ready..... saving to train_dataset")
            os.makedirs(os.path.dirname(self.data_injection.train_data_path), exist_ok=False)
            df.to_csv(self.data_injection.train_data_path, header=True)
            return self.data_injection.train_data_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def last_week_actual_price(self):
        try:
            logging.info("collecting last weeks actual price") 
            eurusd = yf.Ticker("EURUSD=X")
            last_df= eurusd.history(period="max")
            day, month, year=datetime.now().day, datetime.now().month, datetime.now().year
            date_check = datetime(day=day, month=month, year=year)
            nday=date_check.weekday()
            if nday > 0 or nday < 5:
                actual= last_df.tail(5+nday)
                last_actual= actual[:-nday]
                mpf.plot(last_actual, type='candle', style='charles', volume=True)
                plt.savefig(self.data_injection.pictorial_view)
                last_actual.to_csv(self.data_injection.last_week_actual_price)
                logging.info("last weeks price obtained, updating last week actual price")
                return self.data_injection.last_week_actual_price
            else:
                actual=last_df.tail(5)
                mpf.plot(actual, type='candle', style='charles', volume=True)
                plt.savefig(self.data_injection.pictorial_view)
                actual.to_csv(self.data_injection.last_week_actual_price)
                logging.info("last weeks price obtained, updating last week actual price")
                return self.data_injection.last_week_actual_price
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataInjection()
    obj.last_week_actual_price()
        

