import os
import sys
from datetime import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pdw
import seaborn as sns
import tensorflow as tf
from keras import layers, models
from src.exception import CustomException
from src.logger import logging
from src.components.window_generator import WindowGenerator
from src.utils import save_object
from dataclasses import dataclass
import mplfinance as mpf
import yfinance as yf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, InputLayer, RepeatVector, TimeDistributed, Flatten, ConvLSTM2D
from keras.optimizers import SGD


@dataclass
class ModelTrainerConfig():
    open_model_path=os.path.join('modelhouse', 'open_model.h5')
    high_model_path=os.path.join('modelhouse', 'high_model.h5')
    low_model_path =os.path.join('modelhouse', 'low_model.h5')
    close_model_path=os.path.join('modelhouse', 'close_model.h5')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
    MAX_EPOCHS=20
    BATCH=4 
    
class ModelTrainer():
    def __init__(self):
        self.model_paths=ModelTrainerConfig()
        self.model_path_dic={'Open':[self.model_paths.open_model_path],
                             'High':[self.model_paths.high_model_path],
                             'Low':[self.model_paths.low_model_path],
                             'Close':[self.model_paths.close_model_path]}
    def initiate_window_obj(self):
        try:
            for column, _ in self.model_path_dic.items():
                window=WindowGenerator(label_columns=[column])
                self.model_path_dic[column].append(window)
            logging.info('dataset windows initiated')
            return self.model_path_dic
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_trainnig(self, model_dic):
        try:
            for _, list in model_dic.items():
                model =tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(5, 10), name='convid_2_input'),
                                            tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', name='convid_2'),
                                            tf.keras.layers.MaxPooling1D(pool_size=3, name='max_pooling_1d'),
                                            tf.keras.layers.Flatten(name='flatten_2'),
                                            tf.keras.layers.Dense(10, activation='relu', name='dense_4'),
                                            tf.keras.layers.Dense(5, name='dense_5')])
                logging.info('model has been initiated')
                model.compile(loss=tf.keras.losses.MeanSquaredError(),
                              optimizer=tf.keras.optimizers.Adam(),
                              metrics=[tf.keras.metrics.MeanAbsoluteError()])
                logging.info('model compiled')
                model.fit(list[1].train, epochs=self.model_paths.MAX_EPOCHS, callbacks=[self.model_paths.early_stopping], batch_size=self.model_paths.BATCH)
                logging.info('model trained and updated')
                model.save(list[0])
                logging.info('model saved in model.h5 file')
            return 
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=ModelTrainer()
    model_dic=obj.initiate_window_obj()
    obj.initiate_trainnig(model_dic)