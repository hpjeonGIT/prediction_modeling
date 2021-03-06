import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
import time


# source of data: http://www.global-view.com/forex-trading-tools/forex-history/index.html
# https://fred.stlouisfed.org/tags/series?t=oil

# building basic data frame
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
            
        coarse_list.append({'date':ref_date, 'eurusdmonth':avg})
        ref_date = a_date
        local_sum = 0; npt = 0

df_month = pd.DataFrame(coarse_list)
df_month['date'] = pd.to_datetime(df_month['date'], format='%Y-%m')
df_month.head()
plt.gcf()
plt.figure(figsize=(15,4))
plt.plot(df_month['date'],df_month['eurusdmonth'],'o-')
plt.plot(df['date'],df['eurusdclose'])
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
plt.plot(df_month['date'],df_month['eurusdmonth'],'o-')
#plt.xlim(pd.Timestamp('2015-02-15'), pd.Timestamp('2015-07-01'))
#plt.xlim([datetime.date(2010,1,1), datetime.date(2011,1,1)])
plt.show()
#print(type(df_week),type(df_month))

## Building monthly data prediction model
series=df_month['eurusdmonth']
#series = df_week['eurusdweek']
#series = df['eurusdclose']
series_s = series.copy()
window_size=50
for i in range(window_size):
    series = pd.concat([series, series_s.shift(-(i+1))], axis = 1)

series.dropna(axis=0, inplace=True)
nrow = round(0.7*series.shape[0])
train = series.iloc[:nrow, :]; test = series.iloc[nrow:,:]
train_X = train.iloc[:,:-1]; train_y = train.iloc[:,-1]
train_X = train_X.values;    train_y = train_y.values
test_X = test.iloc[:,:-1];   test_y = test.iloc[:,-1]
test_X = test_X.values;      test_y = test_y.values
train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
# Define the LSTM model
model = Sequential()
model.add(LSTM(input_shape = (window_size,1), return_sequences = True, units= window_size))
model.add(Dropout(0.5))
model.add(LSTM(256))  # return a single vector of dimension 32
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(LeakyReLU())
model.add(Activation('linear'))
#opt = SGD(lr=0.01
opt = Adam(lr=0.001)
model.compile(loss="mse", optimizer=opt)
model.summary()

start = time.time()
model.fit(train_X,train_y,batch_size=64,epochs=50,validation_split=0.1, shuffle=False)
print("> Compilation Time : ", time.time() - start)
model_month = model


#  Building daily model using existing data
series = df['eurusdclose']
series_s = series.copy()
window_size=50
series = pd.concat([series, series_s.shift(1)],axis=1)
for i in range(window_size-1):
    series = pd.concat([series, series_s.shift(-(i+1))], axis = 1)

series.dropna(axis=0, inplace=True)
nrow = round(0.7*series.shape[0])
train = series.iloc[:nrow, :]; test = series.iloc[nrow:,:]
train_X = train.iloc[:,:-1]; train_y = train.iloc[:,-1]
train_X = train_X.values;    train_y = train_y.values
test_X = test.iloc[:,:-1];   test_y = test.iloc[:,-1]
test_X = test_X.values;      test_y = test_y.values
train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
# Define the LSTM model
model = Sequential()
model.add(LSTM(input_shape = (window_size,1), return_sequences = True, units= window_size))
model.add(Dropout(0.5))
model.add(LSTM(256))  # return a single vector of dimension 32
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(LeakyReLU())
model.add(Activation('linear'))
#opt = SGD(lr=0.01
opt = Adam(lr=0.001)
model.compile(loss="mse", optimizer=opt)
model.summary()

start = time.time()
model.fit(train_X,train_y,batch_size=64,epochs=50,validation_split=0.1, shuffle=False)
print("> Compilation Time : ", time.time() - start)
model_day = model



### build hybrid model for daily prediction
def find_index_month(date_begin):
    for i in range(len(df_month)):
        tmp = df_month['date'].iloc[i]
        if date_begin.year == tmp.year and date_begin.month == tmp.month:
            break
    i_month = i
    return i_month

nrow = int(0.7*len(df))
window_size = 50
model_day = model
date_begin = df['date'].iloc[nrow]
print('start date is ', date_begin)
i_month = find_index_month(date_begin)
if (i_month < window_size):
    print("Error. i_month is too small")

    
init = i_month; pred_list=[]; ans_list = [];
for i_month in range(init, len(df_month)):
    month_inp = df_month['eurusdmonth'].iloc[i_month-window_size:i_month]
    month_inp = np.array(month_inp)[np.newaxis,:,np.newaxis]
    preds = model_month.predict(month_inp)
    pred_list.append(preds[0][0]); ans_list.append(df_month['eurusdmonth'].iloc[i_month])

plt.plot(pred_list)    
plt.plot(ans_list)
plt.show()

def return_month_pred(i_month):
    month_inp = df_month['eurusdmonth'].iloc[i_month-window_size:i_month]
    month_inp = np.array(month_inp)[np.newaxis,:,np.newaxis]
    preds = model_month.predict(month_inp)
    avg_day = preds[0][0]
    return avg_day

def return_day_pred(i_day, avg_day):
    day_inp = df['eurusdclose'].iloc[i_day-window_size+1:i_day]
    day_inp = day_inp.append(pd.Series([avg_day],index=[i_day])) #mse =0.113
    #day_inp = df['eurusdclose'].iloc[i_day-window_size:i_day] # mse = 0.111
    day_inp = np.array(day_inp)[np.newaxis,:,np.newaxis]
    preds = model_day.predict(day_inp)
    pred_day = preds[0][0]
    return pred_day


old_month = i_month
avg_day = return_month_pred(old_month)
pred_list=[]; ans_list = [];
for i_day in range(nrow,len(df)):
    date_begin = df['date'].iloc[nrow]
    i_month = find_index_month(date_begin)
    if (old_month != i_month):
        avg_day = return_month_pred(i_month)
        old_month = i_month        
    pred_day = return_day_pred(i_day, avg_day)
    pred_list.append(pred_day)
    ans_list.append(df['eurusdclose'].iloc[i_day])
        
plt.plot(pred_list)    
plt.plot(ans_list)
plt.show()

mse = np.mean((np.array(pred_list) - np.array(ans_list)**2))
print(mse)



# low pass filter on eurusd - not recommended at this moment
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

