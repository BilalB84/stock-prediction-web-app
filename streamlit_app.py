import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import LSTM

class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)  # Remove unrecognized argument
        super().__init__(*args, **kwargs)

model = load_model('./app_model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)

tab1, tab2 = st.tabs(["APPLE Stock", "GOOGLE Stock"])

tab1.header('🔮 StockSense AI Web Application')

# Define function to get raw data
def raw_data():
    # Determine end and start dates for dataset download
    end = datetime.now()
    start = datetime(end.year, end.month - 2, end.day)

    # Download Apple's dataset between start and end dates
    apple_df = yf.download('AAPL', start=start, end=end)

    # Rename columns of the Apple DataFrame
    column_dict = {'Open': 'open', 'High': 'high', 'Low': 'low',
                   'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
    apple_df = apple_df.rename(columns=column_dict)
    apple_df.index.names = ['date']
    return apple_df
raw_apple_df = raw_data()

# Define function to calculate 'On Balance Volume (OBV)'
def On_Balance_Volume(Close, Volume):
    change = Close.diff()
    OBV = np.cumsum(np.where(change > 0, Volume, np.where(change < 0, -Volume, 0)))
    return OBV

scaler = MinMaxScaler(feature_range = (0, 1))
def apple_process():
    # Determine end and start dates for dataset download
    end = datetime.now()
    start = datetime(end.year, end.month - 2, end.day)

    # Download Apple's dataset between start and end dates
    apple_df = yf.download('AAPL', start=start, end=end)

    # Rename columns of the Apple DataFrame
    column_dict = {'Open': 'open', 'High': 'high', 'Low': 'low',
                   'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
    apple_df = apple_df.rename(columns=column_dict)
    apple_df.index.names = ['date']

    # Add additional calculated features
    apple_df['garman_klass_volatility'] = ((np.log(apple_df['high']) - np.log(apple_df['low'])) ** 2) / 2 - \
                                          (2 * np.log(2) - 1) * ((np.log(apple_df['adj_close']) - np.log(apple_df['open'])) ** 2)
    apple_df['dollar_volume'] = (apple_df['adj_close'] * apple_df['volume']) / 1e6
    apple_df['obv'] = On_Balance_Volume(apple_df['close'], apple_df['volume'])
    apple_df['ma_3_days'] = apple_df['adj_close'].rolling(3).mean()

    # Filter and preprocess the dataset
    apple_dset = apple_df[['adj_close', 'garman_klass_volatility', 'dollar_volume', 'obv', 'ma_3_days']]
    apple_dset.dropna(axis=0, inplace=True)
    apple_test_scaled = scaler.fit_transform(apple_dset)
    return apple_test_scaled

apple_dataset = apple_process()

def feed_model(dataset, n_past, model, scaler):
    # Create X from the dataset
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i,0])
    testX = np.array(dataX)
    
    # Make predictions using the model
    pred_initial = model.predict(testX)
    
    # Repeat predictions and reshape to original scale
    pred_array = np.repeat(pred_initial, 5, axis = -1)
    preds = scaler.inverse_transform(np.reshape(pred_array, (len(pred_initial), 5)))[:5, 0]
    return preds

prediction = feed_model(apple_dataset, 21, model, scaler).tolist()
# create a dataframe
pred_df = pd.DataFrame({'Predicted Day': ['Tomorrow', '2nd Day', '3rd Day', '4th Day', '5th Day'],
                        'Adj. Closing Price($)': [ '%.2f' % elem for elem in prediction]})

# set the index to the 'name' column
pred_df.set_index('Predicted Day', inplace=True)

# Display result
tab1.col1, tab1.col2 = tab1.columns(2)
with tab1.col1:
    st.subheader('Apple Stock Prediction For Next 5 Days')
    st.write(pred_df)

actual_values  = raw_apple_df['adj_close'].values.tolist()

# Calculate the comparison between predicted next price and last actual price
if actual_values and prediction:
    last_actual_price = actual_values[-1][0]
    next_predicted_price = prediction[0]

    percent_change = (next_predicted_price - last_actual_price) / last_actual_price * 100

    insight = f"""
    <div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6;">
        <strong>The next predicted stock price is:</strong> <span style="color: #4CAF50;">${next_predicted_price:.2f}</span><br>
        <strong>Last actual price:</strong> <span style="color: #FF5722;">${last_actual_price:.2f}</span><br>
        <strong>Change:</strong> <span style="color: {'#4CAF50' if percent_change >= 0 else '#FF5722'};">{percent_change:+.2f}%</span>
    </div>
    """
else:
    insight = "<div style='font-family: Arial, sans-serif;'>Not enough data to generate insights.</div>"

# Display the insight using Markdown with HTML formatting
with tab1.col2:
    st.markdown(insight, unsafe_allow_html=True)

with tab1.expander("AI Model Infographics"):
    multi = '''This project's predictive AI is multivariate LSTM neural networks.  
    Long Short-Term Memory (LSTM) networks, a variant of recurrent neural networks (RNNs), have proven effective for time series forecasting, particularly when dealing with sequential data like stock prices.    
    Stock price movement is influenced by a variety of factors; thus, multivariate time series forecasting is used. The deep learning model captures the underlying patterns and relationships in the data due to domain-based feature engineering.'''
    tab1.markdown(multi)

with tab1.expander("Variables Used by AI"):
    tab1.image('./images/variable-table.png')
    

tab1.markdown(''':rainbow[End-to-end project is done by] :blue-background[Sevilay Munire Girgin]''')
