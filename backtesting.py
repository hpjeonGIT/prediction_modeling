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

# building basic data frame
eurusd = pd.read_csv('eur_usd_hist.csv')
eurusd = eurusd.drop(eurusd.index[0:6]);eurusd.index = range(len(eurusd));eurusd.columns =['date','eurusdclose','high','low']
usdjpn = pd.read_csv('jpn_hist.csv'); 
usdjpn = usdjpn.drop(usdjpn.index[0:6]);usdjpn.index = range(len(usdjpn));usdjpn.columns =['date','usdjpyclose','usdcadclose']
df = pd.merge_ordered(eurusd,usdjpn,on='date')
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


plt.gcf()
plt.figure(figsize=(15,4))
ax = plt.plot(df['date'],df['eurusdclose'],label='eurusd')
ax = plt.plot(df['date'],df['usdcadclose'],label='usdcad')
plt.legend(loc="best")
        
plt.xlim(pd.Timestamp('2007-01-01'), pd.Timestamp('2019-12-01'))    
plt.show()

# Backtesting simulation
# No overhead
def simulate(dx, target_coef):
    ImOn = False
    agg = 0.0; Nentry = 0; Ntarget = 0; Nstop = 0; days_target = []; days_stop = []; date_entry =[]; date_target = []; date_stop = []
    agg_list = []
    # Let's aim target-entry:entry-loss = 3:1
    moving_list_price = []; moving_list_date = []
    N_moving = 30
    for i in range(len(df)):
        today = df ['date'].iat[i]
        price = df['usdcadclose'].iat[i]
        if (ImOn):
            if (price > target_price):
                Ntarget += 1
                ImOn = False
                agg += (price - entry_price)
                agg_list.append({'date':today, 'sum':agg})
                days_target.append(today - day0)
                date_target.append({'date':today, 'price':price})
            if (price < stop_loss):
                Nstop += 1
                ImOn = False
                agg += (price - stop_loss)
                agg_list.append({'date':today, 'sum':agg})
                days_stop.append(today - day0)
                date_stop.append({'date':today, 'price':price})
        if (len(moving_list_price) == N_moving and not ImOn):
            if  (np.mean(moving_list_price) - price) < 0.1*dx:
                Nentry += 1
                entry_price  = price 
                target_price = price + target_coef*dx
                stop_loss    = price - dx
                day0 = today
                ImOn = True
                date_entry.append({'date':today, 'price':target_price})
    
        moving_list_price.append(price)
        moving_list_date.append(today)
        while (len(moving_list_price) > N_moving):
            moving_list_price.pop(0)
            moving_list_date.pop(0)
    return agg, Nentry, Ntarget,Nstop, np.mean(days_target)/np.timedelta64(1,'h'), np.mean(days_stop)/np.timedelta64(1,'h')
        
for coef in [2, 3, 4, 5, 8]:
    for dx in [0.005, 0.01, 0.05, 0.1]:
        ans = simulate(dx, coef); agg = ans[0]; Nentry = ans[1]; Ntarget = ans[2]; Nstop = ans[3]; hour_target = ans[4]; hour_stop = ans[5]
        print('target_coef=%.2f dx=%.4f agg = %.2f norm_agg=%.4f target_stop_ratio =%.2f hour=%4d %4d Nentry = %d\n'%(coef, dx, agg, agg/Nentry, float(Ntarget)/Nstop, hour_target, hour_stop, Nentry))
        
        
        df_entry = pd.DataFrame(date_entry); df_target = pd.DataFrame(date_target); df_stop = pd.DataFrame(date_stop); df_agg = pd.DataFrame(agg_list)
plt.gcf()
plt.figure(figsize=(15,4))
plt.plot(df_entry['date'], df_entry['price'], 'o', markersize=2)
plt.plot(df_target['date'], df_target['price'], 'o',markersize=2,color='green')
plt.plot(df_stop['date'], df_stop['price'], 'o', markersize=2,color='red')
plt.plot(df_agg['date'],df_agg['sum'])
plt.show()
print(not ImOn)
print(len(df))

# Backtesting simulation
# oanda charge rate: $5 for $100,000 trade
# spread cost = (ask - bid)/2 * amount
# assuming 2pip(=0.0002) as fixed spread (not-practical)
Amount = 1000.e2
spread = 0.0002
commission = 5./1.e5
Nstart = 0 #2835
def simulate(dx, target_coef):
    ImOn = False
    agg = Amount; Nentry = 0; Ntarget = 0; Nstop = 0; days_target = []; days_stop = []; date_entry =[]; date_target = []; date_stop = []
    agg_list = []
    # Let's aim target-entry:entry-loss = 3:1
    moving_list_price = []; moving_list_date = []
    N_moving = 30
    for i in range(Nstart,len(df)):
        today = df ['date'].iat[i]
        price = df['eurusdclose'].iat[i]
        if (ImOn):
            if (price > target_price):
                Ntarget += 1
                ImOn = False
                agg = agg + agg*(price - entry_price) - agg*commission - agg*spread
                agg_list.append({'date':today, 'sum':agg})
                days_target.append(today - day0)
                date_target.append({'date':today, 'price':price})
            if (price < stop_loss):
                Nstop += 1
                ImOn = False
                agg = agg + agg*(price - stop_loss) - agg*commission - agg*spread
                agg_list.append({'date':today, 'sum':agg})
                days_stop.append(today - day0)
                date_stop.append({'date':today, 'price':price})
        if (len(moving_list_price) == N_moving and not ImOn):
            if  (np.mean(moving_list_price) - price) < 10*dx:
                Nentry += 1
                entry_price  = price 
                target_price = price + target_coef*dx
                stop_loss    = price - dx
                day0 = today
                ImOn = True
                date_entry.append({'date':today, 'price':target_price})
    
        moving_list_price.append(price)
        moving_list_date.append(today)
        while (len(moving_list_price) > N_moving):
            moving_list_price.pop(0)
            moving_list_date.pop(0)
    if (len(days_target) > 0):
        mean_target = np.mean(days_target)
    else:
        mean_target = pd.Timedelta(1)
    if (len(days_stop) > 0):
        mean_stop = np.mean(days_stop)
    else:
        mean_stop = pd.Timedelta(1)
    
    return agg, Nentry, Ntarget,Nstop, mean_target/np.timedelta64(1,'h'), mean_stop/np.timedelta64(1,'h')

print(df['date'].iat[Nstart],df['date'].iat[-1])
for coef in [0.01, 0.05, 0.1, 0.25, 0.5]:
    for dx in [0.005, 0.01]:
        ans = simulate(dx, coef); agg = ans[0]; Nentry = ans[1]; Ntarget = ans[2]; Nstop = ans[3]; hour_target = ans[4]; hour_stop = ans[5]
        print('target_coef=%.2f dx=%.4f agg = %.2f Ntarget = %d Nstop = %d hour=%4d %4d\n'%(coef, dx, agg, Ntarget, Nstop, hour_target, hour_stop))
        
        
