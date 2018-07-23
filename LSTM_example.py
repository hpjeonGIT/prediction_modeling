import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# building basic data frame
brent = pd.read_csv('crudeOIL_BRENT.csv'); brent.columns = ['date','brent']
wti = pd.read_csv('crudeOIL_WTI.csv'); wti.columns = ['date','wti']
eurusd = pd.read_csv('eur_usd_hist.csv')
eurusd = eurusd.drop(eurusd.index[0:6]);eurusd.index = range(len(eurusd));eurusd.columns =['date','eurusdclose','high','low']
sp500 = pd.read_csv('SP500.csv'); sp500.columns = ['date','sp500']
usdjpn = pd.read_csv('jpn_hist.csv'); 
usdjpn = usdjpn.drop(usdjpn.index[0:6]);usdjpn.index = range(len(usdjpn));usdjpn.columns =['date','usdjpyclose','usdcadclose']
brent.head()
tmp = pd.merge_ordered(brent,wti,on='date')
tmp = pd.merge_ordered(tmp,eurusd,on='date')
tmp = pd.merge_ordered(tmp,usdjpn,on='date')
df = pd.merge_ordered(tmp,sp500,on='date')
df = df.drop(['high','low'],axis=1)
#print(df.head())

# Remove top/bottom NaN
df = df.replace('.',np.nan)
nanMax = 0
for i in df.columns:
    if i != 'date':
        nanMax = max(nanMax,df[i].isna().idxmin())
        print(i,df[i].isna().idxmin())

print(nanMax)
df = df.drop(df.index[0:nanMax])
df.index = range(len(df))
print(df.tail())
nanMin = len(df)
for i in df.columns:
    if i != 'date':
        for n in range(len(df)-1,0,-1):
            if type(df[i].iat[n]) == type('str'):
                print(i,n)
                nanMin = min(nanMin,n)
                break

df = df.drop(df.index[nanMin:])
df.index = range(len(df))
print(df.shape)
df.tail()

# interpolate any missing data        
for i in df.columns:
    if i != 'date':
        df[i] = df[i].astype('float64')
        for n in range(len(df)):
            if np.isnan(df[i].iat[n]):
                for k in range(n+1,len(df)):
                    if ~np.isnan(df[i].iat[k]): # find the closest n+1,2,3 which is not NaN
                        break
                nnext = k
                df[i].iat[n] = (df[i].iat[k] + df[i].iat[n-1])/2. # Just average. Maybe the influence of days might be used
        minx = df[i].min()
        maxx = df[i].max()
        df[i] = (df[i] - minx)/(maxx - minx)
        
for i in df.columns:
    if i != 'date':
        plt.plot(df[i],label=i)
        print(i)
    
plt.show()
