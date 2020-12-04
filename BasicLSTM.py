import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

data = pd.read_csv('GOOG.csv', date_parser = True)
data.tail()

data_training = data[data['Date']<'2018-12-31'].copy()
data_test = data[data['Date']>='2018-12-31'].copy()
# print(data_test)

data_training = data_training.drop(['Date','High','Low'], axis = 1)
# print(data_training)
scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
# print(data_training)

X_train = []
y_train = []

for i in range(20, data_training.shape[0]):
    X_train.append(data_training[i-20:i])
    y_train.append(data_training[i, 0])

data_training=pd.DataFrame(data_training,columns=['Open','Close','Adj Close','Volume'])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape

regressor = Sequential()

# 75's results were good
regressor.add(LSTM(units = 100, activation = 'tanh', recurrent_activation='sigmoid', input_shape = (X_train.shape[1], 4)))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))
regressor.summary()

regressor.compile(optimizer='adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs=66, batch_size=30)


past_60_days = data_training.tail(20)
print(len(data_test))
df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date','High','Low'], axis = 1)
df.head()
print(len(df))
inputs = scaler.transform(df)
inputs

X_test = []
y_test = []
print(inputs.shape[0])
for i in range(20, inputs.shape[0]):
    X_test.append(inputs[i-20:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape
print((len(X_test)))

y_pred = regressor.predict(X_test)
scaler.scale_
scale = 1/scaler.scale_[0]

y_pred = y_pred*scale
y_test = y_test*scale

# Visualising the results
plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real BAJF Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted BAJF Stock Price')
plt.title('BAJFIN Prediction')
plt.xlabel('Time')
plt.ylabel('BAJFIN Real')
plt.legend()
plt.show()
cnt=0
x=0
for i in range(10,len(y_test)):
    print(str(y_pred[i])+" "+str(y_test[i])+" "+str(100-100*y_pred[i]/y_test[i]))
    x+=abs(1-y_pred[i]/y_test[i])
    cnt+=1
print("Average Error is "+str(100*x/cnt))