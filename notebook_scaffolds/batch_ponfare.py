import os, sys 

for i in range(6, 12): 
    iter = i + 1 
    print('writing iter: ' + str(iter)) 
    fo = open('batch_ponfare_iter_'+str(iter)+'.py', 'w') 
    f = open('Master_Scaffold_Ponfare_py3_matching_d2dist_stds.py', 'r') 
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

for i in range(6, 12):
    iter = i + 1
    print('executing iter: ' + str(iter))
    fname = 'batch_ponfare_iter_'+str(iter)+'.py' 
    os.system(f'python {fname}')
