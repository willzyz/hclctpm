import numpy as np, tensorflow as tf, pandas as pd 
import pickle as pkl 

D = pkl.load(open('../data/hscls_seqdata_size754547_allcohorts11weeks_save.pkl', 'r')) 

dgrouped_length = 338465 # after filter: 338461 

datastore = D['datastore'] 
seqlenstore = D['seqlenstore'] 

## fixing length of the data 
datastore = datastore[0:dgrouped_length, :, :] 
datastore = datastore[:, 8:, :] 
seqlenstore = seqlenstore[0:dgrouped_length, 1] 

print('print to verify: ') 
print(datastore[-5:-1, :, :]) 
print(seqlenstore[-5:-1]) 

print('print to verify 0-8 time steps: ') 
print(datastore[0:10, 0:9, :]) 

filter = ~np.logical_and((seqlenstore > 0), (seqlenstore < 8)) 
datastore = datastore[filter, :, :] 
seqlenstore = seqlenstore[filter]

## fixing the length of sequences 
seqlenstore = seqlenstore - 8 

print('assert to verify: ') 
assert(np.sum(seqlenstore < 0) == 0) 
assert(np.sum(seqlenstore == 0) == 0) 


print('final data dimensions: ') 
print(datastore.shape[0]) 
print(len(seqlenstore)) 

import ipdb; ipdb.set_trace 

Dnew = dict() 
Dnew['datastore'] = datastore 
Dnew['seqlenstore'] = seqlenstore 

pkl.dump(Dnew, open('../data/hscls_seqdata_size754547_allcohorts11weeks_save_fixed_new.pkl', 'w')) 
