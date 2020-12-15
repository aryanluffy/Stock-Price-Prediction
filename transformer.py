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
import os,datetime
import warnings
from keract import get_activations
import calendar
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
# import talib

batch_size = 32
seq_len = 20
d_k = 64
d_v = 64
n_heads = 6
ff_dim = 64

class Time2Vector(Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    '''Initialize weights and biases with shape (batch, seq_len)'''
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic = self.add_weight(name='weight_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

  def call(self, x):
    '''Calculate linear and periodic time features'''
    x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
    time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
    time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
    return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
   
  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config

class SingleAttention(Layer):
  def __init__(self, d_k, d_v):
    super(SingleAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v

  def build(self, input_shape):
    self.query = Dense(self.d_k, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')
    
    self.key = Dense(self.d_k, 
                     input_shape=input_shape, 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='glorot_uniform')
    
    self.value = Dense(self.d_v, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')

  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    q = self.query(inputs[0])
    k = self.key(inputs[1])

    attn_weights = tf.matmul(q, k, transpose_b=True)
    attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)
    
    v = self.value(inputs[2])
    attn_out = tf.matmul(attn_weights, v)
    return attn_out    

class MultiAttention(Layer):
  def __init__(self, d_k, d_v, n_heads):
    super(MultiAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.attn_heads = list()

  def build(self, input_shape):
    for n in range(self.n_heads):
      self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
    
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
    self.linear = Dense(input_shape[0][-1], 
                        input_shape=input_shape, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear   

class TransformerEncoder(Layer):
  def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.attn_heads = list()
    self.dropout_rate = dropout

  def build(self, input_shape):
    self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
    self.attn_dropout = Dropout(self.dropout_rate)
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
    self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
    self.ff_dropout = Dropout(self.dropout_rate)
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)

    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer 

  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'d_k': self.d_k,
                   'd_v': self.d_v,
                   'n_heads': self.n_heads,
                   'ff_dim': self.ff_dim,
                   'attn_heads': self.attn_heads,
                   'dropout_rate': self.dropout_rate})
    return config

def create_model():
  '''Initialize time and transformer layers'''
  time_embedding = Time2Vector(seq_len)
  attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  # attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

  '''Construct model'''
  in_seq = Input(shape=(seq_len, 13))
  x = time_embedding(in_seq)
  x = Concatenate(axis=-1)([in_seq, x])
  x = attn_layer1((x, x, x))
  x = attn_layer2((x, x, x))
  # x = attn_layer3((x, x, x))
  x = GlobalAveragePooling1D(data_format='channels_first')(x)
#   x = Dropout(0.1)(x)
#   x = Dense(64)(x)
#   x = Dropout(0.1)(x)
  out = Dense(1)(x)

  model = Model(inputs=in_seq, outputs=out)
  model.compile(loss='mse', optimizer='adam')
  return model




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
    PointSetSize=seq_len+1
    paramtypetostringmap[0]='TimeOfday'
    paramtypetostringmap[1]='Open'
    paramtypetostringmap[2]='High'
    paramtypetostringmap[3]='Low'
    paramtypetostringmap[4]='Close'
    paramtypetostringmap[5]='Volume'
    #how much history to see
    # for j in range(0,1):
    #     paramtypetostringmap[6+3*j]='Low'+str(j)
    #     paramtypetostringmap[7+3*j]='High'+str(j)
    #     paramtypetostringmap[8+3*j]='Close'+str(j)
    paramtypetostringmap[6]='RSI'
    paramtypetostringmap[7]='MACD'
    paramtypetostringmap[8]='CCI'
    paramtypetostringmap[9]='BollingerBandL'
    paramtypetostringmap[10]='BollingerBandU'
    paramtypetostringmap[11]='DayOfWeek'
    paramtypetostringmap[12]='DayOfMonth'
    data = pd.read_csv(ticker+'parameters.csv', date_parser = True)
    data.tail()
    # data=data.drop('TimeOfday',axis=1)
    # data=data.drop('DayOfWeek',axis=1)
    # data=data.drop('DayOfMonth',axis=1)
    # data=data.drop('TimeOfday',axis=1)
    data_training = data[data['Date']<'2020-11-30 09:30:00-05:00'].copy()
    data_test = data[data['Date']>='2020-11-30 09:30:00-05:00'].copy()
    data_training = data_training.drop(['Date'], axis = 1)
    minval=1e9
    #Get min-max for paramtype
    for ind in data_training.index:
        minval=min(minval,data_training[paramtypetostringmap[1]][ind])
    scaler = MinMaxScaler()
    data_training = scaler.fit_transform(data_training)
    # print(data_training)

    X_train = []
    y_train = []
    for i in range(PointSetSize, data_training.shape[0]):
        X_train.append(data_training[i-PointSetSize:i-1])
        y_train.append(data_training[i,1])
    Parameters=[]
    for x in range(0,len(paramtypetostringmap)):
        Parameters.append(paramtypetostringmap[x])
    data_training=pd.DataFrame(data_training,columns=Parameters)

    X_train, y_train = np.array(X_train), np.array(y_train)
    print(X_train.shape[1])
    model = create_model()
    model.summary()
    callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
                                              monitor='val_loss', 
                                              save_best_only=True, verbose=1)

    history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=100, 
                    callbacks=[callback],
                    validation_split=0.1)  

    model = tf.keras.models.load_model('Transformer+TimeEmbedding.hdf5',
                                   custom_objects={'Time2Vector': Time2Vector, 
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder': TransformerEncoder})

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
        y_test.append(inputs[i, 1])

    X_test, y_test = np.array(X_test), np.array(y_test)
    print("kami sama arigato")
    print(len(X_test))
    y_pred=model.predict(X_test)
    scaler.scale_
    scale = 1/scaler.scale_[1]

    y_pred = y_pred*scale+minval
    y_test = y_test*scale+minval
    d_pred=[]
    d_test=[]
    for i in range(60,len(y_pred)):
        d_pred.append(y_pred[i]-y_test[i-1])
        d_test.append(y_test[i]-y_test[i-1])
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
    # y_pred=[2,2,2]
    # y_test=[2,2,2]
    return y_pred,y_test


def main():
    ticker=input("Give Stock name ")
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

