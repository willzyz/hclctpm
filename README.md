### HSCLS: Heterogeneous Sequential Casual Learning System 

Tutorial notes are as below, the comments are given per-script in the following .py scripts: 

ICML entry point: 
For Ponpare data 
notebook_scaffolds/Master_Scaffold_Ponfare_py3_matching_d2dist_otherMs.py 

For USCensus data 
notebook_scaffolds/Master_Scaffold_USCensus_py3_matching_d2dist_otherMs.py 

Entry point: BenchmarkWithCV.py 
--This file runs experiments with Top-Quantile Constraint Ranking, Direct Ranking, and R-learner models, optionally with Causal Forests 

ModelDefinitioins.py 
--This file includes the Tensorflow graph and optimizers, i.e. architecture of models 

dataprep/QueryFunctions.py 
--This file includes the data queries used to pull RxGy, subscriptions, Sequential data using presto queries. 

dataprep/dprepare.py 
--Includes feature and outcome list selection, data normalization through standard gaussian assumptions, removal of nulls, and split of train/validation/test 

dataprep/DataProcFunctions.py 
--Includes scripts to pre-process the data before start to model with TF, R-learner or causal forests 

PlotResults.py
--Plots all plots for experiments to compare TQR, DRM, R-leaner, Duality R-learner, Causal Forests 

Notes on running Causal Forests: 
--grf.R and grf_*.R gives the R code to train causal forests with cost-gain trade-offs 
The package we leaverage is the grf (generalized random forest) code written in R, stable and works with our use-cases and experimentation. 
### https://grf-labs.github.io/grf/
### https://github.com/grf-labs/grf 

FeatureEng.py 
--Feature engineering codes. Includes work to featurize causal models with GBDT tree leaves with r-learner 

LinearHTEModels.py & PromotionModels.py & rlearner.py 
--The codes for R-learner and Heterogeneous Treatement Effect models, needs tidy up from previous code-bases 

Copying Data and previous results: 
ssh username@hadoopgw01-dca1 
hdfs dfs -copyToLocal hdfs://ns-neon-prod-dca1/user/will.zou/hscls/results ~/results
## on local: 
scp -r username@hadoopgw01-dca1:~/results ~/deeplearning_hscls/results 

ssh username@hadoopgw01-dca1 
hdfs dfs -copyToLocal hdfs://ns-neon-prod-dca1/user/will.zou/hscls/data ~/data
## on local: 
scp -r username@hadoopgw01-dca1:~/results ~/deeplearning_hscls/data 

[to update] Notes on Model Datasets: 
- data/LCLatamAug2018TrainingData training data from LuckyCharms project at UberEats, non-sequential 
- data/hscls_seqdata_size754547_allcohorts11weeks_save_fixed_new.pkl training data from Rider-pricing, sequential, 11 weeks, and buckets 0-4, all cohorts 
- data/hscls_seqdata_size1262582_allcohorts11weeks_save_latest.pkl training data from Rider-pricing, sequential, 11 weeks, and all buckets, all cohorts 
- data/hscls_seqdata_size931015_allcohorts2mm11weeks_save.pkl training data from Rider-pricing, sequential, 11 weeks, and all buckets, explore cohort only 

APPENDIX 
---- 
Research notes: Friday August 16th on experimentation, data, model details 

- Eval metric, training objective formulation. 

To either evaluate eventual metric, we would set labels on the last datestr, or last week. This means no matter how long the sequence is, including length 1, we perform evaluatiton only on the last time step. This is different during training. For sequential training, we evaluate and aggregate all the time steps as training objective. Further, the implementation of the this aggregation is as follows, and we start with data definitions. The input to system is a 3-D tensor user x week x features, and we utilize a binary mask of size user x week to indicate whether user was treated that week, the cost and trips data also have same dimensionality user x week. Finally, we utilize a 'sequence-length' variable of length 'user', Thus the objective across all time steps can be computed. Question is how to deal with a non-fixed-length softmax across time steps? Test if this is okay with Tensorflow with masking the rest of dimensions with a very negative number -inf. 

- Deal with missing data. 

To deal with missing data, especially missing time-steps, we can ensure for missing time-steps input the lstm with zero vectors - so that the input doesn't take any effect on recurrency, or we can try learning a missing data vector by back-propagation. this can be easily implemented with Tensorflow. 

- How to benchmark and see improvement of new algorithm? 

Benchmark against DRM on the last time-stamp of each sequence in the test set. To include previous time data, we can add a pooling segment of DRM features, which is average of the previous time-steps, or extend it into a complex spatial pyramid. 

- How to include the treatment label of previous iteration into the next time-step? 

Add in the feature set a label of whether the user was given a promotion in the previous week, if data is missing, give it zero. implement it in the pandas group-by processor function. 

Research notes: Thursday August 22nd on scaling up data 

- Data bias for control and addition of data using DWH 
We extracted three sets of data: 
A. explore only around 420k -> 300k user records 
B. explore + exploit data around 700k -> 380k user records 
C. explore only around 1mm data points -> expect 7-800k user records 
Discovered two problems with data: 
(1) data is shortened because of group by, data placeholder was initialized as number of all records 
(2) the sequence lengths of 11 weeks data are from 8 to 18, should be fixed, causing training issues 

Need to fix the data using scripts and rerun benchmarking on A, B. So the over fitting problem is likely caused by the data issue (1) since validation set is only 30-50k with a few thousands users in control group 

[1:30pm] decided we should fix this data then rerun benchmarks, GPU DWH just got approved: we can start to test GPU instances 
[2:21pm] fixing this data ..... 
[2:53pm] data fixed, found no difference on explore data for drm, seq_start still cannot be shortened, hs model running, moving to all-cohort data 
[2:57pm] both momentum sgd exps are running, running adam as well for hs model and all-cohorts 
[3:02pm] running adam, getting more data for all-cohorts 
[3:15pm] no cpu to run adam. 
[4:02pm] start to get more data on dwh. exp running for 10k iterations, went out for break got usb 
[4:23pm] got data download 
[4:28pm] trying pyspark 
[6:21pm] got 2m data started processing, fixing data found another problem, running 2x 10k steps for 6 unit lstm 
[6:34pm] running data queries 
[11:12pm] got results by normalizing recurrent inputs! better validation cpit by $1, commiting code 
