import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import calendar

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

def dateTimeToTime(datetime):
    x=0.0
    #11,12,14,15
    x=((ord(datetime[11])-ord('0'))*10+ord(datetime[12])-ord('0'))*60+(ord(datetime[14])-ord('0'))*10+ord(datetime[15])-ord('0')
    return x


#paramtype corresponds to open,low,high
def GetPredictions(paramtype,ticker):
    paramtypetostringmap = {}
    paramtypetostringmap[0]='Date'
    paramtypetostringmap[1]='Open'
    paramtypetostringmap[2]='High'
    paramtypetostringmap[3]='Low'
    paramtypetostringmap[4]='Close'
    paramtypetostringmap[5]='Volume'
    paramtypetostringmap[6]='Weekday'
    paramtypetostringmap[7]='DayOfMonth'
    # paramtypetostringmap[8]='VWAP'
    paramtypetostringmap[8]='PrevLow'
    paramtypetostringmap[9]='PrevHigh'
    paramtypetostringmap[10]='PrevClose'
    paramtypetostringmap[11]='RSI'
    daylow={}
    dayhigh={}
    data = pd.read_csv(ticker+'.csv', date_parser = True)
    data.tail()
    # ohdiff=pd.Series([])
    wkday=pd.Series([])
    dayofmonth=pd.Series([])
    vwap=pd.Series([])
    prevlow=pd.Series([])
    prevhigh=pd.Series([])
    prevclose=pd.Series([])
    rsi=pd.Series([])
    vol=0
    weightedsum=0
    currmin=0
    currmax=0
    currclose=0
    for ind in data.index:
        # ohdiff[ind]=data['High'][ind]-data['Open'][ind]
        if data['Date'][ind][11:19]=='09:15:00':
            prevhigh[ind]=currmax
            prevlow[ind]=currmin
            prevclose[ind]=currclose
        else:
            prevlow[ind]=prevlow[ind-1]
            prevhigh[ind]=prevhigh[ind-1]
            prevclose[ind]=prevclose[ind-1]
        if len(prevclose)<13:
            rsi[ind]=0
        else:
            up=0
            down=0
            for j in range(ind-11,ind+1):
                if prevclose[j]-prevclose[j-1]>0:
                    up+=prevclose[j]-prevclose[j-1]
                else:
                    down+=prevclose[j-1]-prevclose[j]
            if data['Close'][ind]-prevclose[ind]>0:
                up+=data['Close'][ind]-prevclose[ind]
            else:
                down-=data['Close'][ind]-prevclose[ind]
            if up+down>0:
                rsi[ind]=up/(up+down)
            else:
                rsi[ind]=0
        currdate=data['Date'][ind][0:10]
        currclose=data['Close'][ind]
        if currdate in daylow:
            daylow[currdate]=min(daylow[currdate],data['Low'][ind])
            dayhigh[currdate]=max(dayhigh[currdate],data['High'][ind])
            currmin=min(currmin,daylow[currdate])
            currmax=max(currmax,dayhigh[currdate])
        else:
            daylow[currdate]=data['Low'][ind]
            dayhigh[currdate]=data['High'][ind]
            currmin=daylow[currdate]
            currmax=dayhigh[currdate]
        if data['Date'][ind][11:19]=='09:15:00':
            vol=data['Volume'][ind]
            weightedsum=data['Volume'][ind]*data['Close'][ind]
        else:
            vol+=data['Volume'][ind]
            weightedsum+=data['Volume'][ind]*data['Close'][ind]
        vwap[ind]=weightedsum/vol
        datestring=data['Date'][ind][0:4]+' '+data['Date'][ind][5:7]+' '+data['Date'][ind][8:10]
        # datestring[4]=' '
        # datestring[7]=' '
        wkday[ind]=findDay(datestring)
        dayofmonth[ind]=(ord(data['Date'][ind][8])-ord('0'))*10+ord(data['Date'][ind][9])-ord('0')
        # print(str(ind)+' '+str(ohdiff[ind]))
    data.insert(6,paramtypetostringmap[6],wkday)
    data.insert(7,paramtypetostringmap[7],dayofmonth)
    # data.insert(8,paramtypetostringmap[8],vwap)
    data.insert(8,paramtypetostringmap[8],prevlow)
    data.insert(9,paramtypetostringmap[9],prevhigh)
    data.insert(10,paramtypetostringmap[10],prevclose)
    data.insert(11,paramtypetostringmap[11],rsi)
    data_training = data[data['Date']<'2020-11-30 09:30:00-05:00'].copy()
    data_test = data[data['Date']>='2020-11-30 09:30:00-05:00'].copy()
    # print("Training Data")
    # print(data)
    # print("Testing Data")
    # print(data_test)
    data_training_dates=pd.Series([])
    for ind in data_training.index:
        data_training_dates[ind]=data_training['Date'][ind]
        data_training['Date'][ind]=dateTimeToTime(data_training['Date'][ind])
    # data_training = data_training.drop(['Date'], axis = 1)
    minval=1e9
    #Get min-max for paramtype
    for ind in data_training.index:
        minval=min(minval,data_training[paramtypetostringmap[1]][ind])
    # print("THE MINVAL IS:: "+str(minval))
    # print(data_training)
    scaler = MinMaxScaler()
    data_training = scaler.fit_transform(data_training)
    # print(data_training)

    X_train = []
    y_train = []
    # print("THE SHAPE OF TRAINING IS::")
    # print(data_training.shape)
    for i in range(15, data_training.shape[0]):
        # print("ESORAGOTO")
        # for j in range(0,5):
        #     print(str(data_training[i][j]/scaler.scale_[j]))
        temp=data_training[i-15:i-1]
        # if temp[10][0]<0.2:
        #     continue
        X_train.append(temp)
        y_train.append(data_training[i, 1])
    Parameters=[]
    for x in range(0,len(paramtypetostringmap)):
        Parameters.append(paramtypetostringmap[x])
    data_training=pd.DataFrame(data_training,columns=Parameters)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train.shape

    regressor = Sequential()
    # simple early stopping
    # 75's results were good
    # regressor.add(LSTM(units = 60,activation = 'tanh',recurrent_activation='sigmoid', return_sequences=True ,input_shape = (X_train.shape[1], 7)))    
    # regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 100,activation = 'sigmoid',recurrent_activation='tanh', input_shape = (X_train.shape[1], len(paramtypetostringmap))))
    # regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.summary()

    regressor.compile(optimizer='adam', loss = 'mean_squared_error')
    es = EarlyStopping(monitor='loss',patience=100,restore_best_weights=True)
    #55,100,80 were good 
    regressor.fit(X_train, y_train, epochs=70, batch_size=30,callbacks=[es])
    # regressor.fit(X_train, y_train, epochs=100, batch_size=30)

    past_60_days = data_training.tail(15)
    # print(len(data_test))
    for ind in data_test.index:
        data_test['Date'][ind]=dateTimeToTime(data_test['Date'][ind])
    df = past_60_days.append(data_test, ignore_index = True)
    
    # df = df.drop(['Date'], axis = 1)
    df.head()
    # print(len(df))
    inputs = scaler.transform(df)
    inputs

    X_test = []
    y_test = []
    for i in range(15, inputs.shape[0]):
        temp=inputs[i-15:i-1]
        # if temp[10][0]<0.2:
        #     continue
        X_test.append(temp)
        y_test.append(inputs[i,1])
        # print("FUCKKKKKK "+str(inputs[i,5]))

    X_test, y_test = np.array(X_test), np.array(y_test)

    y_pred = regressor.predict(X_test)
    scaler.scale_
    scale = 1/scaler.scale_[1]

    y_pred = y_pred*scale+minval
    y_test = y_test*scale+minval

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
    return y_pred,y_test

def GetInputs(paramtype,ticker):
    data = pd.read_csv(ticker+'.csv', date_parser = True)
    data.tail()

    data_training = data[data['Date']<'2020-12-04 09:30:00-05:00'].copy()
    data_test = data[data['Date']>='2020-12-04 09:30:00-05:00'].copy()

    data_training = data_training.drop(['Date'], axis = 1)
    scaler = MinMaxScaler()
    data_training = scaler.fit_transform(data_training)
    data_training=pd.DataFrame(data_training,columns=['Open','High','Low','Close','Volume'])
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
    # getIntradayData(ticker)
    # y_pred_low=GetPredictions(2,ticker)
    # y_pred_high=GetPredictions(1,ticker)
    # y_test_low=GetInputs(2,ticker)
    # y_test_high=GetInputs(1,ticker)
    y_pred,y_test=GetPredictions(0,ticker)
    # for i in range(0,9):
    #     y_pred_temp,y_test=GetPredictions(0,ticker)
    #     y_pred=y_pred+y_pred_temp
    # y_pred=y_pred/10
    for i in range(0,len(y_test)):
        print(str(y_pred[i])+' '+str(y_test[i]))
    mse=0
    # y_pred_close=GetPredictions(3,ticker)
    # y_test_open=GetInputs(0,ticker)
    # y_test_close=GetInputs(3,ticker)
    file_pred=open(ticker+' '+'Open'+'Price'+'Predictions'+'.csv','w+')
    file_pred.write("Actual,Predicted\n")
    for i in range(0,len(y_pred)):
        file_pred.write(str(float(y_test[i]))+','+str(float(y_pred[i]))+'\n')
    file_pred.close()
    returns=0

    # for i in range(10,len(y_test_low)):
    #     # print(str(y_pred_low[i])+str(y_pred_high[i])+str(y_test_low[i])+str(y_test_high[i]))
    #     if y_pred_low[i]>y_test_low[i] and y_pred_high[i]<y_test_high[i]:
    #         returns+=100*(1-y_pred_low[i]/y_pred_high[i])

    variance=0.0
    avg=0
    cnt=0
    for i in range(20,len(y_pred)):
        mse=mse+(y_pred[i]-y_test[i])*(y_pred[i]-y_test[i])
        avg=avg+y_test[i]
        cnt+=1
    avg=avg/cnt
    for i in range(20,len(y_pred)):
        variance=(y_test[i]-avg)*(y_test[i]-avg)+variance
    print("r^2 is")
    print(1-mse/variance)

    # print("The returns are "+str(returns))
    return

main()

