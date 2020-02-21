import numpy as np, tensorflow as tf, pandas as pd 
import pickle as pkl 

D = pkl.load(open('../data/hscls_seqdata_size432955_exploreonly19weeks_save.pkl', 'r')) 

dgrouped_length = 338005 

datastore = D['datastore'] 
seqlenstore = D['seqlenstore'] 

## fixing length of the data 
datastore = datastore[0:dgrouped_length, :, :] 
seqlenstore = seqlenstore[0:dgrouped_length, 1] 

print('print to verify: ') 
print(datastore[-5:-1, :, :]) 
print(seqlenstore[-5:-1]) 

## verify the length of sequences 
import ipdb; ipdb.set_trace() 

print('assert to verify: ') 
assert(np.sum(seqlenstore < 0) == 0) 
assert(np.sum(seqlenstore == 0) == 0) 

Dnew = dict() 
Dnew['datastore'] = datastore 
Dnew['seqlenstore'] = seqlenstore 

pkl.dump(Dnew, open('../data/hscls_seqdata_size432955_exploreonly19weeks_save_fixed_new.pkl', 'w')) 
