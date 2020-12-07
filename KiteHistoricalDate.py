from datetime import datetime,timedelta
from kiteconnect import KiteConnect
import pdb
import pandas as pd

ak='5t4uaet9jycxgcmu'
asecret='cydpv9svuw31wblnvxtt9b8xa1dlxcsy'

kite=KiteConnect(api_key=ak)
tkn
data=kite.generate_session(tkn,api_secret=asecret)
kite.set_access_token(data["access token"])

trd={:{'token':}}