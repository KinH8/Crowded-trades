# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:55:56 2022

@author: Wu
"""
import os
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
# Data cleaning and structuring
temp = os.listdir()
temp = [x for x in temp if '.py' not in x]

for x,y in enumerate(temp):
    name = y.split('.')[0]
    raw = pd.read_csv(y, index_col='Date', parse_dates=True)
    
    if x==0:
        px = raw['PX_LAST'].rename(name).to_frame()
        pb = raw['PX_TO_BOOK_RATIO'].rename(name).to_frame()
        cap = raw['CUR_MKT_CAP'].rename(name).to_frame()
    else:
        px = px.join(raw['PX_LAST'].rename(name), how='outer')
        pb = pb.join(raw['PX_TO_BOOK_RATIO'].rename(name), how='outer')
        cap = cap.join(raw['CUR_MKT_CAP'].rename(name),how='outer')
        
px.to_csv('px.csv')
pb.to_csv('pb.csv')
cap.to_csv('cap.csv')
'''

pxa = pd.read_csv('px.csv',index_col='Date',parse_dates=True)
pb = pd.read_csv('pb.csv',index_col='Date',parse_dates=True)
#cap = pd.read_csv('cap.csv',index_col='Date',parse_dates=True)

px = pxa.pct_change().dropna()

def centrality_score(px):
    pca = PCA(n_components=5)
    pca.fit(px)
    AR = pca.explained_variance_ratio_
    AR = AR.reshape(len(AR),1)
    
    # Formula below based on equation (2) from Kinlaw, Kritzman and Turkington (2019)
    c = np.abs(pca.components_)  # abs(EV(i,j)) where i = sector, j = eigenvector
    c_ = np.ones([c.shape[1],1])
    total = c.dot(c_)
    y = np.divide(c,total)  # abs(EV(i,j)) / cross sectional sum of EV
    
    y2 = np.multiply(AR,y).T  # Multiply with ARj where j = eigenvector
    y3 = y2.sum(axis=1)
    y4 = y3/np.sum(AR)  # each number equals i sector's centrality score
    return y4

centrality = []

lookback = 250
for b in range(lookback,len(px)):
    temp = px.iloc[b-lookback:b]
    centrality.append(centrality_score(temp.values))

centrality = pd.DataFrame(centrality, columns=px.columns,index=px.index[lookback:])
#centrality.plot().legend(loc='center left',bbox_to_anchor=(1.0,0.5)) # stackoverflow how to put legend outside the plot with pandas

centrality.index = centrality.index.date
ax = sns.heatmap(centrality.T, robust=True, cmap='YlGnBu')
plt.figure()

# RELATIVE VALUE
pb = pb.ffill()
valuation = pb/pb.rolling(250).mean(skipna=True)
p = valuation.mean(axis=1, skipna=True)
normalized_pb = valuation.div(p, axis=0)
#normalized_pb.plot().legend(loc='center left',bbox_to_anchor=(1.0,0.5)) # stackoverflow how to put legend outside the plot with pandas

# Plot specific sector
print(px.columns)
sector = 'tech'  # input desired sector
h = centrality[[sector]]
h = h.join(normalized_pb[sector].rename('P/B'), how='left')

fig, ax1 = plt.subplots()
plt.xticks(rotation=45)

ax2 = ax1.twinx()
ax1.plot(h.index, h[sector],'g-')
ax2.plot(h.index,h['P/B'],'b-')

ax1.set_xlabel('Date')
ax1.set_ylabel('Centrality score',color='g')
ax2.set_ylabel('P/B',color='b')
plt.title(sector)