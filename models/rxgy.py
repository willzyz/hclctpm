import os
import sys
import json
import pandas as pd
import numpy as np
import pickle as pkl
import requests
import tensorflow as tf

sys.path.append('../dataprep/')
sys.path.append('../')

from DataProcFunctions import LoadDataFromPkl
from michelangelo import api
from models.ModelDefinitions import *
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from pywebhdfs.webhdfs import PyWebHdfsClient

num_optimize_iterations = 300  ## number of optimization iterations
num_modeling_inits = 2  ## number of random initializations
num_hidden = 0  ## number of hidden units in DNN
save_cf_data = False  ### whether to save data for causal forest training

## set a random seed to reproduce results
seed = 1234;
np.random.seed(seed)  # tf.compat.v2.random.set_seed(seed);

sample_frac = 1.0  ## option to sample data by a fraction \in (0, 1)
data_filename = '/home/udocker/deeplearning_hscls/data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3'
prefix = 'rxgy_v5_07_08_featuremod3_tr_iter100_run4'
PETASTORM_HDFS_DRIVER = 'libhdfs'

def prepare_features(args):
    train_df_path = args.train_data_path
    test_df_path = args.test_data_path
    print("train_df_path: {}".format(train_df_path))
    print("test_df_path: {}".format(test_df_path))

    # with make_batch_reader(train_df_path, hdfs_driver=PETASTORM_HDFS_DRIVER) as train_reader:
    #     with make_batch_reader(test_df_path, hdfs_driver=PETASTORM_HDFS_DRIVER) as test_reader:
    #         train_ds = make_petastorm_dataset(train_reader)
    #         test_ds = make_petastorm_dataset(test_reader)
    #         print("Training data sample: {}".format(train_ds.take(1)))
    #         print("Test data sample: {}".format(test_ds.take(1)))

    train_data_path = '/user/wmingshi/cl_data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3'
    # train_data_path = '/user/wmingshi/cl_data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_feature_lit' # small train dataset

    hdfs = PyWebHdfsClient(host='hadoopneonnamenode02-dca1', port='50070', user_name='michelangelo')
    train_data = hdfs.read_file(train_data_path)

    with open('./data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3', 'wb') as file:
        file.write(train_data)

def train_drm_model(args):
    s3_key_for_model = args.s3_key_for_model

    D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromPkl(
        data_filename, frac=sample_frac, use_python3=True, save_cf_data=save_cf_data)

    print('### ----- start the training of deep learning models ------ ')
    gs_drm = []
    for i in range(num_modeling_inits):
        gs_drm.append(tf.Graph())

    print('------> Training DRM ranking model .... ')
    sess_list = []
    val_results = []
    for i in range(num_modeling_inits):
        print('---> running cross validation, iteration: ' + str(i))
        ### ---- train cpit ranking model for comparison ---
        dobjc, doptc, ddumh, ddumu = DirectRankingModelDNN(gs_drm[i], D_tre, D_unt, o_tre, o_unt, c_tre, c_unt,
                                                           'train-first-drm', num_hidden)

        dsess = tf.Session(graph=gs_drm[i])
        sess_list.append(dsess)

        ### initialize variables and run optimization
        with gs_drm[i].as_default() as g:
            dinit = tf.global_variables_initializer()
        dsess.run(dinit)
        for step in range(num_optimize_iterations):
            _, dobjres = dsess.run([doptc, dobjc])
            if step % 100 == 0:
                print('opt. step : ' + str(step) + ' obj: ' + str(dobjres))

        print('---> optimization finished ... ')

        ### evaluate CPIT metric on validation set
        dobjv, ddumo, dumh, dumhu = DirectRankingModelDNN(gs_drm[i], Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt,
                                                          'eval', num_hidden)
        val_result = dsess.run(dobjv)
        print('validation CPIT:')
        print(val_result)
        val_results.append(val_result)

    best_index = min(enumerate(val_results), key=itemgetter(1))[0]

    print('INFO: best performing model: iteration ' + str(best_index))
    print("Upload model to S3: {}".format(s3_key_for_model))


def train_tqr_model(args):
    s3_key_for_model = args.s3_key_for_model
    ### RxGy TQR setting:
    p_quantile = 0.4  ## percentage of quantile to aim for
    use_schedule = True  ## option to use a constraint annealing schedule
    temp = 0.5  ## initial temperature for constraints
    inc_temp = 0.1  ## increment of temperature per 100 iterations

    D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromPkl(
        data_filename, frac=sample_frac, use_python3=True, save_cf_data=save_cf_data)

    print('### ----- start the training of deep learning models ------ ')
    gs_tqr = []
    for i in range(num_modeling_inits):
        gs_tqr.append(tf.Graph())

    print('------> Training TQR ranking model .... ')
    val_results = []
    sess_list = []
    for i in range(num_modeling_inits):
        print('---> running cross validation, iteration: ' + str(i))
        obj, opt, dumh, dumhu, vtemp, p_quantile = TunableTQRankingModelDNN(gs_tqr[i], D_tre, D_unt, o_tre, o_unt,
                                                                            c_tre, c_unt, 'train-first', temp,
                                                                            p_quantile, num_hidden, use_schedule)
        ### session definitions and variable initialization
        sess = tf.Session(graph=gs_tqr[i])
        sess_list.append(sess)

        ### initialize variables and run optimization
        with gs_tqr[i].as_default() as g:
            init = tf.global_variables_initializer()
        sess.run(init)
        cur_temp = temp
        for step in range(num_optimize_iterations):
            _, objres = sess.run([opt, obj])
            if step % 100 == 0:
                cur_temp = cur_temp + inc_temp
                print('opt. step : ' + str(step) + ' obj: ' + str(objres))
                if use_schedule:
                    sess.run(vtemp.assign(cur_temp))
                    print('setting temperature to :' + str(sess.run(vtemp)))

        print('---> optimization finished ... ')
        tempvalue = sess.run(vtemp)
        p_quantilevalue = p_quantile
        print('temp:')
        print(tempvalue)
        print('p_quantile:')
        print(p_quantilevalue)

        ### evaluate CPIT metric on validation set
        objv, dumo, dumh, dumhu, dvtemp, dp_quantile = TunableTQRankingModelDNN(gs_tqr[i], Dv_tre, Dv_unt, ov_tre,
                                                                                ov_unt, cv_tre, cv_unt, 'eval', temp,
                                                                                p_quantile, num_hidden, use_schedule)

        val_result = sess.run(objv)
        print('validation CPIT:')
        print(val_result)
        val_results.append(val_result)

    from operator import itemgetter
    best_index = min(enumerate(val_results), key=itemgetter(1))[0]

    print("INFO: best performing model: iteration {}".format(str(best_index)))

    ### run scoring on whole test set
    with gs_tqr[best_index].as_default() as g:
        if num_hidden > 0:
            with tf.variable_scope("tqrhidden") as scope:
                h1_test = tf.contrib.layers.fully_connected(Dt, num_hidden, activation_fn=tf.nn.tanh,
                                                            reuse=tf.AUTO_REUSE, scope=scope,
                                                            weights_initializer=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope("tqranker") as scope:
                h_test = tf.contrib.layers.fully_connected(h1_test, 1, activation_fn=None, reuse=tf.AUTO_REUSE,
                                                           scope=scope)
        else:
            with tf.variable_scope("tqranker") as scope:
                h_test = tf.contrib.layers.fully_connected(Dt, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope)
        top_quantile_ranking_scores = sess_list[best_index].run(h_test)[:, 0]

    print("INFO: evaluation of test set completed")
    print("Upload model to S3: {}".format(s3_key_for_model))

    return sess_list, best_index, top_quantile_ranking_scores, ot, wt, ct


def upload_to_s3(data, target_filename):
    s3_key = 'projects/WorkflowTest/tm20191210-002841-XNTMARPF/' + os.path.basename(target_filename)
    api.set_params(email="michelangelo@uber.com", service='cl_workflow', ldap_group="michelangelo")
    resp = api.MichelangeloAPI.generatePresignedURL(s3_key, api.ClientMethodForPresign.PUT)

    requests.put(resp.fileUrl, data=json.dumps(data), headers={'Content-Type': 'binary/octet-stream', 'Accept': 'application/json'})
    print("INFO: resource has been uploaded to s3: {}".format(s3_key))

def train(args):
    prepare_features(args)
    train_tqr_model(args)
