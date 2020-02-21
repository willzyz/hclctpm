import tensorflow as tf, numpy as np, pandas as pd, pickle as pkl 
from ModelDefinitions import * 
from DataProcFunctions import * 
from TPRFunction import * 

## script to run sequential causal learning model 
## optimization settings 
mb_batch_size = 13000 ## use large minibatches 
num_optimize_iterations = 2000 
num_benchmark_iterations = 1000 
dropout_ratio = 0.5 
prepared_data_filename = 'data/hscls_seqdata_size754547_allcohorts11weeks_save_fixed_new.pkl' 
#hscls_seqdata_size1262582_allcohorts11weeks_save_latest.pkl' #hscls_seqdata_size754547_allcohorts11weeks_save_fixed_new.pkl' 
datastr = prepared_data_filename.split('/')[1].split('seqdata')[1].split('save')[0] 
sess = tf.Session() 

tf.set_random_seed(5678) 

## prepare data using dataprocfunctions 
rldata = SeqDrmProcessData(prepared_data_filename, 
                           num_train = 300000, ##num_train=700000, 
                           seq_start=0) 
batchedDataset = rldata[0]; valDataFeatures = rldata[1]; valDataSeqLens = rldata[2]
valDataTreatmentLabels = rldata[3]; valDataTripsLabels = rldata[4]; valDataCostLabels = rldata[5] 
valDataGBLabels = rldata[6]; valDataExistLabels = rldata[7]

print('----> verifying data dimensions... --------') 
num_weeks = valDataFeatures.shape[1] 
feature_dims = valDataFeatures.shape[2] 
print('number of weeks: ' + str(num_weeks) + '  feature_dim: ' + str(feature_dims)) 
print(' ') 

num_units = [16, 16] 
model_def = ''
for n in num_units: 
    model_def = model_def + '_' + str(n) 
model_def = model_def + '_' 

## tensor place-holders for tensorflow graph 
trainDataFeatures_ph = tf.placeholder(tf.float64, shape=(None, num_weeks, feature_dims)) 
trainDataSeqLens_ph = tf.placeholder(tf.float64, shape=(None, )) 
trainDataTreatmentLabels_ph = tf.placeholder(tf.float64, shape=(None, num_weeks)) 
trainDataTripsLabels_ph = tf.placeholder(tf.float64, shape=(None, num_weeks)) 
trainDataCostLabels_ph = tf.placeholder(tf.float64, shape=(None, num_weeks)) 
trainDataGBLabels_ph = tf.placeholder(tf.float64, shape=(None, num_weeks)) 
trainDataExistLabels_ph = tf.placeholder(tf.float64, shape=(None, num_weeks)) 

## define model by building a tf graph 
cpitobj, cpitopt, cpigbobj, cpigbopt, dumh_rscore = HetSequentialCausalModel(num_units, 
                                                                             trainDataFeatures_ph, 
                                                                             trainDataSeqLens_ph, 
                                                                             trainDataTreatmentLabels_ph, 
                                                                             trainDataTripsLabels_ph, 
                                                                             trainDataCostLabels_ph, 
                                                                             trainDataGBLabels_ph, 
                                                                             trainDataExistLabels_ph, 
                                                                             'training') 

"""
### initialize variables and run optimization 
init = tf.global_variables_initializer(); sess.run(init) 

### start optimization with minibatch adam 
agg = 0 ## running average 
saver = tf.train.Saver() 
print('----> starting optimization ---------') 
print('using running average objective') 
for step in range(num_optimize_iterations + 2): 
    rlmb = PullMinibatch(batchedDataset, mb_batch_size) 
    _, objres = sess.run([cpitopt, cpitobj], 
                         feed_dict={trainDataFeatures_ph:rlmb[0], 
                                    trainDataSeqLens_ph:rlmb[1], 
                                    trainDataTreatmentLabels_ph:rlmb[2], 
                                    trainDataTripsLabels_ph:rlmb[3], 
                                    trainDataCostLabels_ph:rlmb[4], 
                                    trainDataGBLabels_ph:rlmb[5], 
                                    trainDataExistLabels_ph: rlmb[6]}) 
    agg = agg + objres 
    meanagg = agg * 1.0 / (step + 1) 
    if step % 100 == 0: 
        print('st:' + str(step) + '-obj:' + "%.3f" % meanagg + '  ,'), 
    if step % 200 == 0: 
        ## save model 
        save_path = "modelsaves/hscls_"+str(model_def)+"_numiter_" + str(num_optimize_iterations) + "_step_" + str(step) + "_dropout_" + str(dropout_ratio) + "_data_" +str(datastr) + "_model.ckpt" 
        save_path = saver.save(sess, save_path) 
        print(' ') 
        print("Model saved in path: %s" % save_path) 

print(' ') 
print('----> optimization finished ---------') 
print(' ') 
"""
saver = tf.train.Saver() 
saver.restore(sess, "modelsaves/hscls__16_16__numiter_10000_step_9800_data__size754547_allcohorts11weeks__model.ckpt")
#hscls__16_16__numiter_2000_step_1800_dropout_0.5_data__size754547_allcohorts11weeks__model.ckpt") 

vcpitobj, vcpitopt, vcpigbobj, vcpigbopt, vhsclsrnkscores = HetSequentialCausalModel(num_units, 
                                                                                     valDataFeatures, 
                                                                                     valDataSeqLens, 
                                                                                     valDataTreatmentLabels, 
                                                                                     valDataTripsLabels, 
                                                                                     valDataCostLabels, 
                                                                                     valDataGBLabels, 
                                                                                     valDataExistLabels, 
                                                                                     'validating')

vcpitobj_res = sess.run(vcpitobj) 
print('validation cpit: ' + str(vcpitobj_res)) 
hsclsvalscores = sess.run(vhsclsrnkscores) 
hsclsvalscores = np.reshape(hsclsvalscores, (-1,)) 
hsclsvalscores = hsclsvalscores[np.reshape(valDataExistLabels, (-1,)) > 0.5] 

### call benchmarking function 
TPRFunction(hsclsvalscores, num_benchmark_iterations) 
