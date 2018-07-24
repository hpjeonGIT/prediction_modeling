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
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
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
        
plt.gcf()
plt.figure(figsize=(15,4))
for i in df.columns:
    if i != 'date':
        ax = plt.plot(df['date'],df[i],label=i)
        plt.legend(loc="best")
        print(i)        
        
plt.xlim(pd.Timestamp('2007-01-01'), pd.Timestamp('2019-12-01'))    
plt.show()

# average per month
ref_date = df['date'].iloc[0]
local_sum = 0; npt = 0 
coarse_list = []
for i in range(1,len(df)):
    a_date = df['date'].iloc[i]
    if ((ref_date.year == a_date.year) and (ref_date.month == a_date.month)):
        local_sum += df['eurusdclose'].iloc[i]
        npt += 1
    else:
        if npt > 0:
            avg = local_sum/npt
        else:
            avg = 0
            
        coarse_list.append({'date':str(a_date.year)+'-'+str(a_date.month), 'eurusdmonth':avg})
        ref_date = a_date
        local_sum = 0; npt = 0

df_month = pd.DataFrame(coarse_list)
df_month['date'] = pd.to_datetime(df_month['date'], format='%Y-%m')
df_month.head()
plt.plot(df_month['date'],df_month['eurusdmonth'])
plt.show()

# average per week
ref_date = df['date'].iloc[0]
local_sum = 0; npt = 0 
coarse_list = []
for i in range(1,len(df)):
    a_date = df['date'].iloc[i]
    if ((ref_date.year == a_date.year) and (ref_date.week == a_date.week)):
        local_sum += df['eurusdclose'].iloc[i]
        npt += 1
    else:
        if npt > 0:
            avg = local_sum/npt
        else:
            avg = 0
            
        if (avg > 0):
            coarse_list.append({'date':ref_date, 'eurusdweek':avg})
        ref_date = a_date
        local_sum = 0; npt = 0

df_week = pd.DataFrame(coarse_list)
df_week['date'] = pd.to_datetime(df_week['date'], format='%Y-%U')
df_week.head()
plt.gcf()
plt.figure(figsize=(15,4))
plt.plot(df_week['date'],df_week['eurusdweek'],'o-')
plt.plot(df['date'],df['eurusdclose'])
#plt.xlim(pd.Timestamp('2015-02-15'), pd.Timestamp('2015-07-01'))
#plt.xlim([datetime.date(2010,1,1), datetime.date(2011,1,1)])
plt.show()

# low pass filter on eurusd - not recommended
from scipy import fftpack
sig_fft = fftpack.fft(df['eurusdclose'])
power = np.abs(sig_fft)
sample_freq = fftpack.fftfreq(len(df), d=1)
#plt.plot(sample_freq,power); plt.show()
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]
print(peak_freq)
plt.gcf()
plt.figure(figsize=(5,4))
plt.plot(df['eurusdclose'], label='Original signal')
anslist = [0,1,2]
for i,j in zip([0.001, 0.01, 0.1], [0,1,2]):
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > i] = 0
    filtered_sig = fftpack.ifft(high_freq_fft)
    plt.plot(filtered_sig, linewidth=3, label="criterion ="+str(i))
    plt.legend(loc="best")
    anslist[j] = filtered_sig
    
plt.show()
plt.figure(figsize=(5,4))
plt.plot(df_month['date'],df_month['eurusdmonth'])
plt.plot(df['date'],anslist[2])
plt.show()
