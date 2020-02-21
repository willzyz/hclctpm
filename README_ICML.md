### Supplementary Material for ICML paper: Learning Continuous Treatment Policy and Bipartite Embeddings for Matching with Heterogeneous Causal Effects 
Paper ID: 3127 

Please find in the attached .zip file: 
(1) all required files to compile latex for the paper (starting from main.txt in the root folder) 
(2) codes for experiments in the paper. This is stored in the folder: CODE_for_icml/ 

The tutorial notes for the code are as below, the comments are given per-script in the following .py scripts: 
[run with python 3 (tested with python 3.6) and tensorflow code with tensorflow 1.15] 

ICML entry point: 
For Ponpare data, Entry point
notebook_scaffolds/Master_Scaffold_Ponfare_py3_matching_d2dist_otherMs.py 

For USCensus data, Entry point 
notebook_scaffolds/Master_Scaffold_USCensus_py3_matching_d2dist_otherMs.py 

icml_vis_std*.py 
--Plots all plots for experiments to compare Simple TC models, R-leaner, Duality R-learner, Causal Forests 

icml_ctpm_analysis.ipynb 
--Code for producing embedding t-sne visualization, and scatterplot plots in the analysis section 

Entry point: BenchmarkWithCV.py 
--This file runs experiments with Top-Quantile Constraint Ranking, Direct Ranking, and R-learner models, optionally with Causal Forests 

models/ModelDefinitioins.py 
--This file includes the Tensorflow graph and optimizers, i.e. architecture of models 

dataprep/QueryFunctions.py 
--This file includes the data queries used to pull RxGy, subscriptions, Sequential data using presto queries. 

dataprep/dprepare.py 
--Includes feature and outcome list selection, data normalization through standard gaussian assumptions, removal of nulls, and split of train/validation/test 

dataprep/DataProcFunctions.py 
--Includes scripts to pre-process the data before start to model with TF, R-learner or causal forests 

Notes on running Causal Forests: 
--causaltree/grf.R and grf_*.R gives the R code to train causal forests with cost-gain trade-offs 
The package we leaverage is the grf (generalized random forest) code written in R, stable and works with our use-cases and experimentation. 
### https://grf-labs.github.io/grf/
### https://github.com/grf-labs/grf 

LinearHTEModels.py & PromotionModels.py & rlearner.py 
--The codes for R-learner and Heterogeneous Treatement Effect models, needs tidy up from previous code-bases 
