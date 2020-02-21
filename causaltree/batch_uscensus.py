import os, sys 

for i in range(10): 
    iter = i + 1 
    print('writing iter: ' + str(iter)) 
    fo = open('batch_grf_uscensus_ctpm_d2dist_iter_'+str(iter)+'.R', 'w') 
    f = open('grf_uscensus_ctpm_d2dist.R', 'r') 
    fo.write('global_iter = ' + str(iter)) 
    fo.write('\n') 
    line = 1 
    while line: 
        line = f.readline() 
        if not line: 
            break 
        fo.write(line) 
    fo.close() 
    f.close() 

for i in range(10): 
    iter = i + 1 
    print('executing iter: ' + str(iter)) 
    fname = 'batch_grf_uscensus_ctpm_d2dist_iter_'+str(iter)+'.R'
    os.system(f'Rscript {fname}') 
