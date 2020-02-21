library(grf) 
library(dplyr) 
library(data.table) 

#suffix = 'finalsize' 
prefix = 'rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3' 
num_trees = 3 
p_alpha = 0.2 #0.05 
p_min_node_size = 3 #10 
p_sample_fraction = 0.5 #0.8 
num_features = 39 

print('reading data from csv') 
dread <- read.csv(file=paste('causal_forest_r_data_', prefix, '_train.csv', sep='')) 

print('performing data transformations') 
features <- select(dread, 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',  'V7', 'V8', 'V9', 'V10', 'V11', 'V12',  'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39') 
w <- select(dread, 'treatment') 
o <- select(dread, 'V40') 
c <- select(dread, 'cost') 

X <- matrix(as.vector(t(features)), , num_features, byrow = TRUE)
W<- matrix(as.vector(t(w)), , 1)
Y<- matrix(as.vector(t(o)), , 1)
C<- matrix(as.vector(t(c)), , 1) 

print('fitting causal random forest for trips') 
tauO.forest <- causal_forest(X, Y, W, num.trees=num_trees, alpha=p_alpha, min.node.size=p_min_node_size, sample.fraction=p_sample_fraction) #tune.parameters=TRUE) #
#tauO.forest$tuning.output 

print('fitting causal random forest for cost') 
tauC.forest <- causal_forest(X, C, W, num.trees=num_trees, alpha=p_alpha, min.node.size=p_min_node_size, sample.fraction=p_sample_fraction) #tune.parameters=TRUE) #
#tauC.forest$tuning.output 

print('reading test data from csv, data transformations') 
dread <- read.csv(file=paste('causal_forest_r_data_', prefix,'_test.csv')) 

features <- select(dread, 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',  'V7', 'V8', 'V9', 'V10', 'V11', 'V12',  'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39') 
w <- select(dread, 'treatment') 
o <- select(dread, 'V40') 
c <- select(dread, 'cost') 

Xtest <- matrix(as.vector(t(features)), , num_features, byrow = TRUE) 

print('performing prediction') 
tauO.hat <- predict(tauO.forest, Xtest) 
tauC.hat <- predict(tauC.forest, Xtest) 

print('predicting and writing to csv, order') 
tdfO <- as.data.frame(t(tauO.hat)) 
write.csv(tdfO, file = paste('results/causal_forest_grf_test_set_results_O_',prefix, '_numtrees', toString(num_trees), '_alpha', str(p_alpha), '_min_node_size', str(p_min_node_size), '_sample_fraction', str(p_sample_fraction),'.csv', sep='')) 

print('predicting and writing to csv, cost') 
tdfC <- as.data.frame(t(tauC.hat)) 
write.csv(tdfC, file = paste('results/causal_forest_grf_test_set_results_C_',prefix, '_numtrees', toString(num_trees), '_alpha', str(p_alpha), '_min_node_size', str(p_min_node_size), '_sample_fraction', str(p_sample_fraction),'.csv', sep='')) 
