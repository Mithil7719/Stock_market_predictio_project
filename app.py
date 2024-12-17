import numpy as np
import pandas as pd
# NumPy and Pandas are used for data manipulation and calculations.
import yfinance as yf
# yfinance (yf) is used to download historical stock data from Yahoo Finance.
from keras.models import load_model
# Keras is used to load a pre-trained deep learning model (Stock Predictions Model.keras), which will predict stock prices.
import streamlit as st
# Streamlit (st) is used to create an interactive web app.
import matplotlib.pyplot as plt
import plotly.express as px
# Matplotlib and Plotly are used for visualizations (charts).


model = load_model('C:\stock\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symnbol', 'AAPL')
start = '2012-01-01'
end = '2022-12-21'

data = yf.download(stock, start ,end)

# fig = px.line(data, x = data.index, y = data['Adj Close'], title=stock)
# st.plotly_chart(fig)

st.subheader('Stock Data')
# st.write(data)

pricing_data, news = st.tabs(["Pricing Data", "Top 10 News"])
with pricing_data:
    st.write('Price Movements')
    data2 = data
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace = True)
    st.write(data2)
    annual_return = data2['% Change'].mean()*252*100
    st.write('Annual Return is ', annual_return,'%')
    stdev = np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation is ',stdev*100,'%')
    st.write('Risk Adj. Return is ',annual_return/(stdev*100))
    
#     from stocknews import StockNews    
# with news:
#     st.header(f'News of {stock}')
#     sn = StockNews(stock, save_news=False)
#     df_news = sn.read_rss()
#     for i in range(10):
#         st.subheader(f'News {i+1}')
#         st.write(df_news['published'][i])
#         st.write(df_news['title'][i])
#         st.write(df_news['summary'][i])
#         title_sentiment = df_news['sentiment_title'][i]
#         st.write(f'Title Sentiment {title_sentiment}')
#         news_sentiment = df_news['sentiment_summary'][i]
#         st.write(f'News Sentiment {news_sentiment}')
# The app has two tabs: Pricing Data and Top 10 News. Currently, the news section is commented out, but it would display news related to the stock.


data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
# The MinMaxScaler is used to scale the stock data to a range between 0 and 1, which is commonly required for neural network models.


pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label = "50 Days Moving Avg")
plt.plot(data.Close, 'g', label = "Original Price")
plt.show()
st.pyplot(fig1)
# st.plotly_chart(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label = "50 Days Moving Avg")
plt.plot(ma_100_days, 'b', label = "100 Days Moving Avg.")
plt.plot(data.Close, 'g', label = "Original Price")
plt.show()
st.pyplot(fig2)
# st.plotly_chart(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label = "100 Days Moving Avg.")
plt.plot(ma_200_days, 'b', label = "200 Days Moving Avg.")
plt.plot(data.Close, 'g', label = "Original Price")
plt.show()
st.pyplot(fig3)
# st.plotly_chart(fig3)
# The app visualizes the stock price with various Moving Averages (MA):
# 50-Day Moving Average (MA50)

# 100-Day Moving Average (MA100)
# 200-Day Moving Average (MA200)
# These are plotted using Matplotlib, with the stock price shown alongside these averages.


x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time', color = 'white')
plt.ylabel('Price', color = 'white')
plt.show()
st.pyplot(fig4)
# st.plotly_chart(fig4)