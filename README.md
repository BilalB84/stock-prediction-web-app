# SafeStock AI Web Application 🔮
## Real-Time Stock Forecasting & Analysis Using LSTM Models 🤖
SafeStock AI is a web-based application that combines real-time stock market data with AI-driven forecasting to provide actionable insights on stocks.   
It leverages LSTM (Long Short-Term Memory) models for time-series forecasting, offering users predictions and technical analysis for a better understanding of market trends.


## Demo App

[![SafeStock AI Web App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://safestock-ai.streamlit.app/)


## Key Features
📈 Real-Time Stock Prices: Stay updated with dynamic market data from Yahoo Finance.
🤖 AI Predictions: View the next 5-day stock price forecasts, powered by advanced LSTM models.
📊 Technical Analysis: Explore trends with indicators like SMA, EMA, RSI, and OBV using interactive charts.
📋 Performance Metrics: Dive into AI model accuracy with evaluation details and key variables.


## How It Works
The app is powered by:

-  Data Preprocessing: Historical stock data is cleaned, scaled, and prepared for analysis.
- LSTM Model: A robust time-series forecasting algorithm trained to predict stock price movements.
- Visualization: Interactive charts display historical trends, technical indicators, and model predictions.

## Screenshots of Web App
[<img src="https://github.com/SevilayMuni/stock-prediction-web-app/blob/master/images/tab1-ss.png" width="600"/>](https://github.com/SevilayMuni/stock-prediction-web-app/blob/master/images/tab1-ss.png)
[<img src="https://github.com/SevilayMuni/stock-prediction-web-app/blob/master/images/tab2-ss.png" width="600"/>](https://github.com/SevilayMuni/stock-prediction-web-app/blob/master/images/tab2-ss.png)
[<img src="https://github.com/SevilayMuni/stock-prediction-web-app/blob/master/images/tab3-ss.png" width="600"/>](https://github.com/SevilayMuni/stock-prediction-web-app/blob/master/images/tab3-ss.png)

## Models & Methodology
The core of SafeStock AI is an LSTM neural network, specifically designed for sequential data like time series. The model is trained on historical stock prices and technical indicators to predict future movements.

## Model Features
Input Variables: 
    
    Garman-Klass Volatility
    Dollar Volume
    On Balance Volume (OBV)
    Moving Average Convergence Divergence (MACD)
    Moving Averages (MAs)

Evaluation Metrics:
    
    RMSE (Root Mean Squared Error)
    MAE (Mean Absolute Error)
    R² (Coefficient of Determination)

Technical Indicators:
  
    SMA (Simple Moving Average): Tracks short-term trends.
    EMA (Exponential Moving Average): Highlights momentum.
    RSI (Relative Strength Index): Analyzes overbought/oversold conditions.
    OBV (On-Balance Volume): Measures buying/selling pressure.

## Future Enhancements
- Add support for more stocks and global markets.
- Incorporate additional AI models like GRU and ARIMA for comparative analysis.
- Deploy for scalability on cloud platforms (AWS, GCP, or Azure).
- Provide user-defined customization for technical indicators.

## Disclaimer
❗ SafeStock AI is a tool for research and educational purposes only. It does not provide financial or investment advice.

## Contact 📩
For any questions or inquiries, feel free to reach out:
- **LinkedIn:** [Sevilay Munire Girgin](www.linkedin.com/in/sevilay-munire-girgin-8902a7159)
Thank you for visiting my project repository. Happy and accurate predicting! 💕
