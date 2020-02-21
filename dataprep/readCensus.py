import os, sys
import pandas as pd, numpy as np, tensorflow as tf 

f = open('USCensus1990.data.txt', 'r') 

cnt = 0 

l = f.readline() 
col_names = l.split(',') 
col_names = [a.strip() for a in col_names] 
print(col_names) 
data = pd.DataFrame(columns=col_names) 
print(data) 

while 1: 
    l = f.readline()
    if not l: 
        break 
    #print(l) 
    if cnt % 10000 == 0: 
        print(cnt) 
    if cnt > 200000: 
        break 
    cnt = cnt + 1     
    values =[float(a.strip()) for a in l.split(',')]
    data = data.append(dict(zip(col_names, values)), ignore_index=True)

data.to_csv('USCensus1990_data.csv') 

### 
# use income as reward outcome 
# use 
