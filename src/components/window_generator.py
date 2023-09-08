import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class WindowGeneratorConfig():
    dataset_file_path=os.path.join('dataset', 'train_dataset.csv')

class WindowGenerator():
    def __init__(self, df_path='dataset/train_dataset.csv', label_columns=None):
        self.window_generator=WindowGeneratorConfig()
        logging.info("store the raw data in a dataframe")
        self.df=pd.read_csv(df_path)
        # Work out the label column indices.
        logging.info("working out the label column indices")
        self.label_columns = label_columns
        if self.label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
            self.column_indices = {name: i for i, name in enumerate(self.df.columns)}
        # Work out the window parameters.
        logging.info("working out window parameters")
        self.input_width = 5
        self.label_width = 5
        self.shift = 1
        self.total_window_size = self.input_width + self.shift
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        try:
            logging.info("start slice dataset into windows")
            inputs = features[:, self.input_slice, :]
            labels = features[:, self.labels_slice, :]
            if self.label_columns is not None:
                labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],axis=-1)
                # Slicing doesn't preserve static shape information, so set the shapes
                # manually. This way the `tf.data.Datasets` are easier to inspect.
            inputs.set_shape([None, self.input_width, None])
            labels.set_shape([None, self.label_width, None])
            logging.info("window slicing complete")
            return inputs, labels
        except Exception as e:
            raise CustomException(e, sys)
    
    def make_dataset(self):
        try:
            self.data = np.array(self.df, dtype=np.float32)
            self.dataset = tf.keras.utils.timeseries_dataset_from_array(data=self.data, targets=None, 
                                                              sequence_length=self.total_window_size,
                                                              sequence_stride=1, shuffle=True,  batch_size=32,)
            logging.info("data has been converted to tensorflow dataset")
            self.dataset= self.dataset.map(self.split_window)
            return self.dataset
        except Exception as e:
            raise CustomException(e,sys)
        
    @property
    def train(self):
        try:
            return self.make_dataset()
        except Exception as e:
            raise CustomException(e, sys)



if __name__ =="__main__":
    obj=WindowGenerator(label_columns=['Open'])
    dataset= obj.train
    print(dataset.element_spec)


    




    

