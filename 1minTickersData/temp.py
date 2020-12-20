import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ISSE_NAHI_HUA_KUCH_BHI_TOH_CHODO OpenPricePredictions.csv")

y_pred=[]
y_test=[]

for ind in data.index:
    if ind==0:
        continue
    y_test.append(100*data['Actual'][ind]-100*data['Actual'][ind-1])
    y_pred.append(100*data['Predicted'][ind]-100*data['Actual'][ind-1])

plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Stock Price')
plt.title(' Prediction')
plt.xlabel('Time')
plt.ylabel(' Real')
plt.legend()
plt.show()