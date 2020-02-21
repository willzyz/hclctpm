import os, sys 

for i in range(12, 18): 
    iter = i + 1 
    print('writing iter: ' + str(iter)) 
    fo = open('batch_uscensus_iter_'+str(iter)+'.py', 'w') 
    f = open('Master_Scaffold_USCensus_py3_matching_d2dist_std.py', 'r') 
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

for i in range(12, 18): 
    iter = i + 1 
    print('executing iter: ' + str(iter)) 
    fname = 'batch_uscensus_iter_'+str(iter)+'.py' 
    os.system(f'python {fname}') 
