library(grf) 
library(dplyr) 
library(data.table) 

suffix = 'tiny' 
num_trees = 100 

print('reading data from csv') 
dread <- read.csv(file=paste('causal_forest_r_data_train_',suffix,'.csv', sep='')) 

print('performing data transformations') 
features <- select(dread, 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',  'V7', 'V8', 'V9', 'V10', 'V11', 'V12',  'V13', 'V14', 'V15', 'V16', 'V17', 'V18')
w <- select(dread, 'treatment')
o <- select(dread, 'V19') 
c <- select(dread, 'cost') 

X <- matrix(as.vector(t(features)), , 18, byrow = TRUE)
W<- matrix(as.vector(t(w)), , 1)
Y<- matrix(as.vector(t(o)), , 1)
C<- matrix(as.vector(t(c)), , 1) 

print('fitting causal random forest for trips') 
tauO.forest <- causal_forest(X, Y, W, num.trees=num_trees, tune.parameters=TRUE) #alpha=0.2, min.node.size=50, sample.fraction=0.5, 
tauO.forest$tuning.output 

print('fitting causal random forest for cost') 
tauC.forest <- causal_forest(X, C, W, num.trees=num_trees, tune.parameters=TRUE) #alpha=0.2, min.node.size=50, sample.fraction=0.5, 
tauC.forest$tuning.output 

print('reading test data from csv, data transformations') 
dread <- read.csv(file='causal_forest_r_data_test.csv')

features <- select(dread, 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',  'V7', 'V8', 'V9', 'V10', 'V11', 'V12',  'V13', 'V14', 'V15', 'V16', 'V17', 'V18') 
w <- select(dread, 'treatment') 
o <- select(dread, 'V19') 
c <- select(dread, 'cost') 

Xtest <- matrix(as.vector(t(features)), , 18, byrow = TRUE) 

print('performing prediction') 
tauO.hat <- predict(tauO.forest, Xtest) 
tauC.hat <- predict(tauC.forest, Xtest) 

print('predicting and writing to csv, order') 
tdfO <- as.data.frame(t(tauO.hat)) 
#write.csv(tdfO, file = paste('results/causal_forest_grf_test_set_results_O_',suffix, '_numtrees', toString(num_trees), '.csv', sep='')) 
fwrite(tdfO, paste('results/causal_forest_grf_test_set_results_O_',suffix, '_numtrees', toString(num_trees), '2.csv', sep=''), verbose=TRUE) 

print('predicting and writing to csv, cost') 
tdfC <- as.data.frame(t(tauC.hat)) 
#write.csv(tdfC, file = paste('results/causal_forest_grf_test_set_results_C_',suffix, '_numtrees', toString(num_trees), '.csv', sep='')) 
fwrite(tdfC, paste('results/causal_forest_grf_test_set_results_C_',suffix, '_numtrees', toString(num_trees), '2.csv', sep=''), verbose=TRUE) 
