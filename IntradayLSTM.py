import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def getIntradayData(ticker):
    data=yf.download(ticker,period='60d',interval='5m',auto_adjust='True')
    print(data)
    file=open(ticker+'.csv','w+')
    file.write('Date,Open,High,Low,Close,Adj Close,Volume\n')
    for ind in data.index:
        file.write(str(ind)+','+str(data['Open'][ind])+','+str(data['High'][ind])+','+str(data['Low'][ind])+','+str(data['Close'][ind])+',0,'+str(data['Volume'][ind])+'\n')
    file.close()
    return 


#paramtype corresponds to open,low,high
def GetPredictions(paramtype,ticker):
    data = pd.read_csv(ticker+'.csv', date_parser = True)
    data.tail()

    data_training = data[data['Date']<'2020-12-03 15:58:00-05:00'].copy()
    data_test = data[data['Date']>='2020-12-03 15:58:00-05:00'].copy()
    print(data_test)

    data_training = data_training.drop(['Date'], axis = 1)
    # print(data_training)
    scaler = MinMaxScaler()
    data_training = scaler.fit_transform(data_training)
    # print(data_training)

    X_train = []
    y_train = []

    for i in range(20, data_training.shape[0]):
        X_train.append(data_training[i-20:i])
        y_train.append(data_training[i, paramtype])

    data_training=pd.DataFrame(data_training,columns=['Open','High','Low','Close','Adj Close','Volume'])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train.shape

    regressor = Sequential()
    # simple early stopping
    # 75's results were good
    regressor.add(LSTM(units = 100,activation = 'tanh',recurrent_activation='sigmoid', input_shape = (X_train.shape[1], 6)))

    regressor.add(Dense(units = 1))
    regressor.summary()

    regressor.compile(optimizer='adam', loss = 'mean_squared_error')
    es = EarlyStopping(monitor='loss',patience=20,restore_best_weights=True)
    #55,100 were good 
    regressor.fit(X_train, y_train, epochs=110, batch_size=30,callbacks=[es])
    # regressor.fit(X_train, y_train, epochs=100, batch_size=30)

    past_60_days = data_training.tail(20)
    print(len(data_test))
    df = past_60_days.append(data_test, ignore_index = True)
    df = df.drop(['Date'], axis = 1)
    df.head()
    print(len(df))
    inputs = scaler.transform(df)
    inputs

    X_test = []
    y_test = []
    for i in range(20, inputs.shape[0]):
        X_test.append(inputs[i-20:i])
        y_test.append(inputs[i, paramtype])

    X_test, y_test = np.array(X_test), np.array(y_test)

    y_pred = regressor.predict(X_test)
    scaler.scale_
    scale = 1/scaler.scale_[paramtype]

    y_pred = y_pred*scale
    y_test = y_test*scale

    # Visualising the results
    plt.figure(figsize=(14,5))
    plt.plot(y_test, color = 'red', label = 'Real Stock Price')
    plt.plot(y_pred, color = 'blue', label = 'Predicted Stock Price')
    plt.title(ticker+' Prediction')
    plt.xlabel('Time')
    plt.ylabel(ticker+' Real')
    plt.legend()
    plt.show()
    cnt=0
    x=0
    for i in range(10,len(y_test)):
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
    return y_pred

def GetInputs(paramtype,ticker):
    data = pd.read_csv(ticker+'.csv', date_parser = True)
    data.tail()

    data_training = data[data['Date']<'2019-12-31'].copy()
    data_test = data[data['Date']>='2019-12-31'].copy()

    data_training = data_training.drop(['Date'], axis = 1)
    scaler = MinMaxScaler()
    data_training = scaler.fit_transform(data_training)
    data_training=pd.DataFrame(data_training,columns=['Open','High','Low','Close','Adj Close','Volume'])
    past_60_days = data_training.tail(20)
    df = past_60_days.append(data_test, ignore_index = True)
    df = df.drop(['Date'], axis = 1)
    df.head()
    inputs = scaler.transform(df)
    inputs

    y_test = []
    for i in range(20, inputs.shape[0]):
        y_test.append(inputs[i, paramtype])

    y_test = np.array(y_test)

    scaler.scale_
    scale = 1/scaler.scale_[paramtype]

    y_test = y_test*scale

    return y_test

def main():
    ticker=input("Give Stock name ")
    getIntradayData(ticker)
    y_pred_low=GetPredictions(2,ticker)
    y_pred_high=GetPredictions(1,ticker)
    # y_test_low=GetInputs(2,ticker)
    # y_test_high=GetInputs(1,ticker)
    y_pred_open=GetPredictions(0,ticker)
    y_pred_close=GetPredictions(3,ticker)
    # y_test_open=GetInputs(0,ticker)
    # y_test_close=GetInputs(3,ticker)
    returns=0

    # for i in range(10,len(y_test_low)):
    #     # print(str(y_pred_low[i])+str(y_pred_high[i])+str(y_test_low[i])+str(y_test_high[i]))
    #     if y_pred_low[i]>y_test_low[i] and y_pred_high[i]<y_test_high[i]:
    #         returns+=100*(1-y_pred_low[i]/y_pred_high[i])

    # for i in range(10,len(y_test_open)):
    #     returns+=abs(1-y_test_close[i]/y_test_open[i])

    # print("The returns are "+str(returns))
    return

main()

