import datetime
import time
from kiteconnect import KiteConnect
from kiteconnect import exceptions
import json
import pandas as pd

tdelta0 = datetime.timedelta(days=30)
tdelta = datetime.timedelta(days=29)
interval = "minute"
repeatation = 1

ak = 'mewmepcd9vjx9acz'
asecret = 'dswigcj932ahusvba2uk1iia7j6v6xnx'


kite = KiteConnect(api_key=ak)
# request_tkn = input("[*] Enter Your Request Token Here : ")
# data = kite.generate_session(request_tkn, api_secret=asecret)
# kite.set_access_token(data["access_token"])
# print(data['access_token'])
kite.set_access_token('SlWctga9258A7nd9Tvdxxw9C2PX7FwVo')

savedata = {}

trd_portfolio = {'BAJAJFINSV':{'token':4268801},'ICICIBANK':{'token':1270529},\
                'AXISBANK':{'token':1510401},\
                'HDFCBANK':{'token':341249},\
                'KOTAKBANK':{'token':492033},\
                'L&T':{'token':2939649},\
                'SBIN':{'token':779521},\
                'WIPRO':{'token':969473},\
                'AIRTEL':{'token':2714625},\
                'BANDHANBNK':{'token':579329},\
                'M&MFIN':{'token':3400961}
                }

def myconverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


print("data")

while True:
    for name in trd_portfolio:
        print(name)
        token = trd_portfolio[name]['token']
        curr_month=1
        curr_year=2016
        main_record = []
        towrite=open(str(name)+".csv", "a")
        towrite.write('Date,Open,High,Low,Close,Volume\n')
        while 1:
            next_month=curr_month+1
            next_year=curr_year
            if curr_month==12:
                next_month=1
                next_year+=1
            records = kite.historical_data(token, datetime.date(curr_year,curr_month,2), datetime.date(next_year,next_month,1), '5minute')
            print(len(records))
            if len(records)==0:
                break
            #print("got data",trd_portfolio[token])
            for ind in records:
                towrite.write(str(ind['date'])+','+\
                str(ind['open'])+','+\
                str(ind['high'])+','+\
                str(ind['low'])+','+\
                str(ind['close'])+','+\
                str(ind['volume'])+'\n')
            curr_month=next_month
            curr_year=next_year
            if curr_month==1 and next_year==2021:
                break
            time.sleep(1)
        # print(kk)
        towrite.close()
        # open(str(trd_portfolio[token])+".txt","w+")
    break
