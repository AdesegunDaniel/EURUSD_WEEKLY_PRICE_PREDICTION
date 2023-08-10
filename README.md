# EURUSD_WEEKLY_PRICE_PREDICTION_USING_DEEP_LEARNING
EURUSD Weekly Price Prediction provides analysis and forecasts for the exchange rate between the euro and the US dollar. Features code snippets, charts, and data sources that can help traders, investors make informed decisions are provided in this document. 


# EURUSD Weekly Price Prediction (Summary)

This repository contains the code and data for predicting the weekly price movement of the EURUSD currency pair using machine learning models. The goal is to forecast whether the EURUSD will close higher or lower than the previous week, based on various features such as technical indicators, sentiment analysis, and macroeconomic data.

The main steps of the project are:

- Data collection and preprocessing
- Feature engineering and selection
- Model training and evaluation
- Model deployment and testing

The project is written in Python and uses libraries such as pandas, TensorFlow. The code is organized into modules and scripts that can be run from the command line or a Jupyter notebook. The data is stored in CSV files and can be downloaded from the links provided in the data folder.

The project is still a work in progress but as proven reliable through the moment of monitoring. Any feedback or suggestions are welcome. Please refer to the documentation for more details on how to use the code and data.

# EURUSD Weekly Price Prediction

This is a project that aims to predict the weekly price movement of the EURUSD currency pair using machine learning models. The project is hosted on GitHub and can be accessed at https://.

## Motivation

The EURUSD is one of the most traded and liquid currency pairs in the forex market. It represents the exchange rate between the euro and the US dollar. The price movement of the EURUSD is influenced by various factors, such as economic data, monetary policy, geopolitical events, market sentiment, and technical analysis. Predicting the future price direction of the EURUSD can help traders and investors to make better decisions and optimize their returns.

## Data

The data used for this project consists of daily historical opning and closing price of EURUSD but grouped into weekly historical prices from January 2003 to till date. The data was obtained from https://www.investing.com/currencies/eur-usd-historical-data. The data includes the following features:

- Date: The date of the week
- Price: The closing price of the week
- Open: The opening price of the week
- High: The highest price of the week
- Low: The lowest price of the week
- Change %: The percentage change of the price from the previous week

## Methodology

The project follows these steps:

- Data exploration: Analyzing the data to understand its characteristics and trends
- Data preprocessing: Cleaning and transforming the data to make it suitable for modeling
- Feature engineering: Creating new features from the existing data to capture more information and patterns
- Model selection: Choosing and comparing different machine learning models to find the best one for the task
- Model evaluation: Evaluating the performance of the selected model on unseen data using various metrics
- Model deployment: Deploying the model as a web application using Streamlit

## Results

The best model for this project was cnn_univ_5, which achieved a mean absolute error (MAE) of 0.0056 and a coefficient of determination (R2) of 0.82 on the test set. The model was able to capture the general trend and direction of the EURUSD price, as well as some of the fluctuations and outliers. The model was also able to forecast the price for the next week with reasonable accuracy.

The web application allows users to interact with the model and see its predictions for different weeks. Users can also see the historical prices and the actual vs predicted prices in a line chart. The web application can be accessed at https://eurusd-prediction.herokuapp.com/.

## Future Work

Some possible improvements and extensions for this project are:

- Incorporating more features, such as technical indicators, sentiment analysis, or news headlines
- Using different models, such as neural networks, support vector machines, or gradient boosting
- Tuning the hyperparameters of the models to optimize their performance
- Updating the data and the model regularly to reflect the latest market conditions
