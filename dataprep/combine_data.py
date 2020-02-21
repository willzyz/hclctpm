import numpy as np, pickle as pkl 

D0 = pkl.load(open('../data/hscls_seqdata_size754547_allcohorts11weeks_save_fixed_new.pkl', 'r'))
D1 = pkl.load(open('../data/hscls_seqdata_size169692_allcohorts11weekscap4to5_save.pkl', 'r'))
D2 = pkl.load(open('../data/hscls_seqdata_size169666_allcohorts11weekscap6to7_save.pkl', 'r'))
D3 = pkl.load(open('../data/hscls_seqdata_size168677_allcohorts11weekscap8to9_save.pkl', 'r'))

datastore = np.concatenate((D0['datastore'], D1['datastore']), axis = 0) 
datastore = np.concatenate((datastore, D2['datastore']), axis = 0) 
datastore = np.concatenate((datastore, D3['datastore']), axis = 0) 
import ipdb; ipdb.set_trace() 
seqlenstore = np.concatenate((D0['seqlenstore'], D1['seqlenstore'][:, 1]), axis = 0) 
seqlenstore = np.concatenate((seqlenstore, D2['seqlenstore'][:, 1]), axis = 0) 
seqlenstore = np.concatenate((seqlenstore, D3['seqlenstore'][:, 1]), axis = 0) 

D = dict() 
D['datastore'] = datastore 
D['seqlenstore'] = seqlenstore 

pkl.dump(D, open('../data/hscls_seqdata_size1262582_allcohorts11weeks_save_latest.pkl', 'w')) 


