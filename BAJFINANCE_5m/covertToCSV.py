import pandas as pd

df=pd.read_json('BAJFINANCE.json')
filetowrite=open('BAJFINANCE.csv','a')
# filetowrite.write(str(df))
# for ind in df.index:
#     filetowrite.write(str(df['date'][ind])+','+\
#     str(df['open'][ind])+','+str(df['high'][ind])+','+\
#     str(df['low'][ind])+','+str(df['close'][ind])+','+\
#     str(df['volume'][ind])+'\n')
