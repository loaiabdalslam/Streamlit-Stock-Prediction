from calendar import c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2021-12-31'


st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

st.subheader('Data from 2010-2021')
st.write(df.describe())


#Visualizations
st.subheader('Closing Price vs Time Chart')
st.line_chart(df.Close)


#Visualizations
st.subheader('Closing Price vs Time Chart with 100MA')
mma100 = df.Close.rolling(100).mean()
df.insert(6, "mma100", mma100, True)
st.line_chart(df[['Close','mma100']])


#Visualizations
st.subheader('Closing Price vs Time Chart with 200MA & 100 MA')

mma200 = df.Close.rolling(200).mean()
df.insert(7, "mma200", mma200, True)
st.line_chart(df[['Close','mma100','mma200']])


#Data Splitting 
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
data_testing_array = scaler.fit_transform(data_testing)


x_train = []
y_train = [] 


for i in range(100,data_testing_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])


x_train,y_train = np.array(x_train),np.array(y_train)


#Load My Model 
model = load_model('keras_model.h5')


#Testing Part 
past_100_days= data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_train),np.array(y_train)
y_predicted = model.predict(x_test)

scaler_percent = scaler.scale_
scale_factor =  1/scaler_percent


y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
st.subheader('Acutal Trend vs Predicted Trend')
y_predicted = [ x[0] for x in y_predicted]
dfx = pd.DataFrame(data={'predicted':y_predicted,'actual':y_test})
st.line_chart(dfx)


