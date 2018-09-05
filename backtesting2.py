import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from scipy import signal
from collections import Counter

# building basic data frame
eurusd = pd.read_csv('eur_usd_hist.csv')
eurusd = eurusd.drop(eurusd.index[0:6]);eurusd.index = range(len(eurusd));eurusd.columns =['date','eurusdclose','high','low']
usdjpn = pd.read_csv('jpn_hist.csv'); 
usdjpn = usdjpn.drop(usdjpn.index[0:6]);usdjpn.index = range(len(usdjpn));usdjpn.columns =['date','usdjpyclose','usdcadclose']
df = pd.merge_ordered(eurusd,usdjpn,on='date')
df = df.drop(['high','low'],axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
#print(df.head())

#http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/EURUSD
mindata = pd.read_csv('nov01_07_2011.csv',header=None)
mindata = mindata.drop(mindata.columns[[0, 1, 3,4,5]], axis=1)
mindata.columns = ['date','eurusdclose']
mindata['date'] = mindata.index
mindata['date'].iat[0]

#http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/EURUSD
mindf_2017 = pd.read_csv('DAT_ASCII_EURUSD_M1_2017.csv',header=None, sep=';')
mindf_2017 = mindf_2017.drop(mindf_2017.columns[[0, 1, 3,4,5]], axis=1)
df = mindf_2017
df.columns = ['eurusdclose']

B, A = signal.butter(1, 0.1, output='ba')
C105= signal.filtfilt(B,A, df['eurusdclose'].values)
plt.gcf()
plt.figure(figsize=(15,4))
ax = plt.plot(df['eurusdclose'],label='eurusd', marker='o')
ax = plt.plot(C105,label='1, 0.9')
ax = plt.plot([0,50000,100000],[1.05, 1.05, 1.05],'o')
#ax = plt.plot(df['date'],df['usdcadclose'],label='usdcad')
plt.legend(loc="best")
        
#plt.xlim(pd.Timestamp('2007-01-01'), pd.Timestamp('2007-3-01'))    
#plt.ylim([1.28,1.35])
plt.xlim([000,100000])
#plt.ylim([1.035, 1.045])
plt.show()

# Backtesting simulation
# oanda charge rate: $5 for $100,000 trade
# spread cost = (ask - bid)/2 * amount
# assuming 2pip(=0.0002) as fixed spread (not-practical)
Amount = 10.e0
Nentry = 0; Nlimit=1000
spread = 0.0002; margin = 0.0005
commission = 5./1.e5
N_slide_down = 200; crit_down = -0.0001/N_sliding; crit_up = - crit_down; entry_list = []
N_slide_up = 100;
entry_x = []; entry_y = []; exit_x = []; exit_y = []
N_holding = 500; list_hold = []; slide_down = []; slide_up = []
ImOn = False; agg = Amount*Nlimit; 

for i in range(0, len(df)):
    v_now = df['eurusdclose'].iat[i]
    # 
    # setup past mean
    list_hold.append(v_now)
    if len(list_hold) > N_holding:
        list_hold.pop(0)
    if len(list_hold) == N_holding:
        past_mean = np.mean(list_hold) 
    else:
        past_mean =  -999.
    #
    # moving sliding window
    slide_down.append(v_now); slide_up.append(v_now)
    if len(slide_down) > N_slide_down:
        slide_down.pop(0)
    if len(slide_up) > N_slide_up:
        slide_up.pop(0)
    #
    # Long 
    if (len(slide_down) == N_slide_down and np.mean(slide_down) < past_mean):
        z = np.polyfit(np.linspace(0,N_slide_down-1, N_slide_down), slide_down, 1)
        grad = z[0]; p = np.poly1d(z)
        if (grad <crit_down and Nentry < Nlimit):
            # Let's buy
            entry_list.append(v_now)
            Nentry += 1
            slide_down = slide_down[N_sliding//2:]
            entry_x.append(i); entry_y.append(v_now)
            agg -= Amount
            #print(i, v_now, Nentry)
    # 
    # Short
    if (len(slide_up) == N_slide_up and np.mean(slide_up) > past_mean):
        z = np.polyfit(np.linspace(0,N_slide_up-1, N_slide_up), slide_up, 1)
        grad = z[0]; p = np.poly1d(z)
        if (grad > crit_up and v_now < p(N_slide_up) and Nentry > 0):
            # Let's sell
            for a in entry_list:
                if a + margin < v_now:
                    entry_list.remove(a)
                    Nentry -= 1
                    exit_x.append(i); exit_y.append(v_now)
                    overhead = Amount*commission + Amount*spread
                    profit = Amount*(v_now - a)
                    agg += Amount + (profit - overhead)
                    print(i, v_now, Nentry, profit, overhead, agg)
            
print(agg, Nentry)

# Backtesting simulation
# oanda charge rate: $5 for $100,000 trade
# spread cost = (ask - bid)/2 * amount
# assuming 2pip(=0.0002) as fixed spread (not-practical)
#
# 0903 - 2nd order fitting
df = mindf_2017 #[0:100000]
df.columns = ['eurusdclose']

Amount = 200.e0
Nentry = 0; Nlimit=50
spread = 0.0002; margin = 0.005; stop_crit = 0.02
commission = 5./1.e5
N_slide_down = 30; crit_down = -0.001/N_sliding; crit_up = - crit_down; entry_list = []; 
N_slide_up = 10; 
entry_x = []; entry_y = []; exit_x = []; exit_y = []
N_holding = 500; list_hold = []; slide_down = []; slide_up = []
ImOn = False; agg = Amount*Nlimit; Nlong = 0; Nshort = 0; Nstop = 0

for i in range(0, len(df)):
    v_now = df['eurusdclose'].iat[i]
    # 
    # setup past mean
    list_hold.append(v_now)
    if len(list_hold) > N_holding:
        list_hold.pop(0)
    if len(list_hold) == N_holding:
        past_mean = np.mean(list_hold) 
    else:
        past_mean =  -999.
    #
    # moving sliding window
    slide_down.append(v_now); slide_up.append(v_now)
    if len(slide_down) > N_slide_down:
        slide_down.pop(0)
    if len(slide_up) > N_slide_up:
        slide_up.pop(0)
    #
    # Long 
    if (len(slide_down) == N_slide_down and np.mean(slide_down) < past_mean):
        z = np.polyfit(np.linspace(0,N_slide_down-1, N_slide_down), slide_down, 2)
        grad = z[0]; #print(grad)
        if (grad > 1.e-9 and Nentry < Nlimit):
            # Let's buy
            entry_list.append({'id':i,'v':v_now})
            Nentry += 1
            slide_down = [] #slide_down[N_sliding//2:]
            entry_x.append(i); entry_y.append(v_now)
            agg -= Amount
            Nlong += 1 
            #print(i, v_now, Nentry)
    # 
    # Short
    if (len(slide_up) == N_slide_up and np.mean(slide_up) > past_mean):
        z = np.polyfit(np.linspace(0,N_slide_up-1, N_slide_up), slide_up, 2)
        grad = z[0]; p = np.poly1d(z)
        #
        # Exit
        if (grad < -1e-9 and v_now < p(N_slide_up) and Nentry > 0):
            # Let's sell
            rem_list = []
            for n in range(len(entry_list)):
                if entry_list[n]['v'] + margin < v_now and (i - entry_list[n]['id']) > 10 :
                    Nentry -= 1
                    exit_x.append(i); exit_y.append(v_now)
                    overhead = Amount*commission + Amount*spread
                    profit = Amount*(v_now - entry_list[n]['v'])
                    agg += Amount + (profit - overhead)
                    rem_list.append(n)
                    Nshort += 1 
                    if (Nentry < 1):
                        print("%d %.2f %.4f %d %.3f  %.3f %.1f"%(i, i/len(df), v_now, Nentry, profit, overhead, agg))
            for n in reversed(rem_list):
                entry_list.pop(n)
    #
    # Stop
    rem_list = []
    for  n in range(len(entry_list)):
        if entry_list[n]['v'] - stop_crit > v_now :
            Nentry -= 1
            exit_x.append(i); exit_y.append(v_now)
            overhead = Amount*commission + Amount*spread
            loss = Amount*(v_now - entry_list[n]['v'])
            agg += Amount + (loss - overhead)
            rem_list.append(n)
            Nstop += 1
            print("stop loss %d %.2f %.4f %d %.3f  %.3f %.1f"%(i, i/len(df), v_now, Nentry, profit, overhead, agg))
    for n in reversed(rem_list):
        entry_list.pop(n)
            
print(agg, Nentry, Nlong, Nshort, Nstop)
print("finalizing")
for n in range(len(entry_list)):
    Nentry -= 1
    exit_x.append(i); exit_y.append(v_now)
    overhead = Amount*commission + Amount*spread
    loss = Amount*(v_now - entry_list[n]['v'])
    agg += Amount + (loss - overhead)
    Nstop += 1
print(agg, Nentry, Nlong, Nshort, Nstop)

plt.gcf()
plt.figure(figsize=(15,4))
plt.plot(df['eurusdclose'])
plt.plot(entry_x, entry_y, 'o')
plt.plot(exit_x, exit_y, 'o')
#plt.xlim([0,10000])


