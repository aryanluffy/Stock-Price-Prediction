import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras.losses
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from attention import Attention
import os
from keract import get_activations
import calendar
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
# import talib


def findDay(date): 
    year, month, day = (int(i) for i in date.split(' '))     
    dayNumber = calendar.weekday(year, month, day) 
    return dayNumber 

def getIntradayData(ticker):
    data=yf.download(ticker,period='60d',interval='5m',auto_adjust='True')
    print(data)
    file=open(ticker+'.csv','w+')
    file.write('Date,Open,High,Low,Close,Volume\n')
    for ind in data.index:
        file.write(str(ind)+','+str(data['Open'][ind])+','+str(data['High'][ind])+','+str(data['Low'][ind])+','+str(data['Close'][ind])+','+str(data['Volume'][ind])+'\n')
    file.close()
    return 

def sign_penalty(y_true, y_pred):
    penalty = 1
    loss = tf.where(tf.less(y_true * y_pred, 0), \
                     penalty * tf.square(y_true - y_pred), \
                     tf.square(y_true - y_pred))

    return tf.reduce_mean(loss, axis=-1)


#paramtype corresponds to open,low,high
def GetPredictions(paramtype,ticker):
    paramtypetostringmap = {}
    PointSetSize=20
    # paramtypetostringmap[0]='Date'
    for i in range(0,134):
        paramtypetostringmap[i]='feature_'+str(i+1)
    data = pd.read_csv(ticker, date_parser = True)
    data.tail()
    data_training = data[data['Date']<'2020-01-16 09:15:00+05:30'].copy()
    data_test = data[data['Date']>='2020-01-16 09:15:00+05:30'].copy()
    data_training = data_training.drop(['Date'], axis = 1)
    minval=1e9
    #Get min-max for paramtype
    for ind in data_training.index:
        minval=min(minval,data_training['feature_101'][ind])
    scaler = MinMaxScaler()
    data_training = scaler.fit_transform(data_training)
    # print(data_training)

    X_train = []
    y_train = []
    for i in range(PointSetSize, data_training.shape[0]):
        X_train.append(data_training[i-PointSetSize:i-1])
        y_train.append(data_training[i,100])
    Parameters=[]
    for x in range(0,len(paramtypetostringmap)):
        Parameters.append(paramtypetostringmap[x])
    data_training=pd.DataFrame(data_training,columns=Parameters)

    X_train, y_train = np.array(X_train), np.array(y_train)
    print(X_train.shape[1])
    regressor = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], len(paramtypetostringmap)), return_sequences=True),
        Attention(name='attention_weight'),
        Dense(1, activation='linear')
    ])
    regressor.summary()

    regressor.compile(optimizer='adam', loss = 'mean_squared_error')
    es = EarlyStopping(monitor='val_loss',patience=1000,restore_best_weights=True)
    #30 were good 
    regressor.fit(X_train, y_train,validation_split=0.002,epochs=100, batch_size=30,callbacks=[es])
    past_60_days = data_training.tail(PointSetSize)
    # print(len(data_test))
    df = past_60_days.append(data_test, ignore_index = True)
    
    df = df.drop(['Date'], axis = 1)
    df.head()
    # print(len(df))
    inputs = scaler.transform(df)
    inputs

    X_test = []
    y_test = []
    for i in range(PointSetSize, inputs.shape[0]):
        X_test.append(inputs[i-PointSetSize:i-1])
        y_test.append(inputs[i, 100])

    X_test, y_test = np.array(X_test), np.array(y_test)
    print("kami sama arigato")
    print(len(X_test))
    y_pred = regressor.predict(X_test)
    scaler.scale_
    scale = 1/scaler.scale_[100]

    y_pred = y_pred*scale+minval
    y_test = y_test*scale+minval
    d_pred=[]
    d_test=[]
    for i in range(0,len(y_pred)):
        d_pred.append(1000*y_pred[i]-1000*y_test[i-1])
        d_test.append(1000*y_test[i]-1000*y_test[i-1])
    # Visualising the results
    plt.figure(figsize=(14,5))
    plt.plot(d_test, color = 'red', label = 'Real Stock Price')
    plt.plot(d_pred, color = 'blue', label = 'Predicted Stock Price')
    plt.title(ticker+' Prediction')
    plt.xlabel('Time')
    plt.ylabel(ticker+' Real')
    plt.legend()
    plt.show()
    cnt=0
    x=0
    for i in range(0,len(y_test)):
        # print(str(y_test[i])+str(y_pred[i]))
        x+=abs(1-y_pred[i]/y_test[i])
        cnt+=1
    if paramtype==0:
        print("Average Error in Opening is "+str(100*x/cnt))
    elif paramtype==1:
        print("Average Error in High is "+str(100*x/cnt))
    elif paramtype==2:
        print("Average Error in Low is "+str(100*x/cnt))
    elif paramtype==3:
        print("Average Error in Close is "+str(100*x/cnt))
    # y_pred=[2,2,2]
    # y_test=[2,2,2]
    return y_pred,y_test


def main():
    # ticker=input("Give Stock name ")
    ticker='ISSE_NAHI_HUA_KUCH_BHI_TOH_CHODO'
    y_pred,y_test=GetPredictions(0,ticker)
    for i in range(0,len(y_test)):
        print(str(y_pred[i])+' '+str(y_test[i]))
    mse=0
    file_pred=open(ticker+' '+'Open'+'Price'+'Predictions'+'.csv','w+')
    file_pred.write("Actual,Predicted\n")
    for i in range(0,len(y_pred)):
        file_pred.write(str(float(y_test[i]))+','+str(float(y_pred[i]))+'\n')
    file_pred.close()
    returns=0
    variance=0.0
    avg=0
    cnt=0
    for i in range(40,len(y_pred)):
        mse=mse+(y_pred[i]-y_test[i])*(y_pred[i]-y_test[i])
        avg=avg+y_test[i]
        cnt+=1
    avg=avg/cnt
    for i in range(40,len(y_pred)):
        variance=(y_test[i]-avg)*(y_test[i]-avg)+variance
    print("r^2 is")
    print(1-mse/variance)
    return
keras.losses.sign_penalty = sign_penalty
main()

