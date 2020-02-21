global_iter = 7
library(grf) 
library(dplyr) 
library(data.table) 

#global_iter = 1 
#suffix = 'finalsize' 
prefix = 'ponfare_v3_causal_data_with_intensity_matching_d2dist' #'data' #r2e_v5_07_08_featuremod3_tr_iter100_run7' 
num_trees = 50 
p_alpha = 0.2 #0.05 
p_min_node_size = 3 #10 
p_sample_fraction = 0.5 #0.8 
num_features = 151 

print('reading data from csv') 
dread <- read.csv(file=paste('causal_forest_r_data_', prefix, '_train.csv', sep='')) 

print('performing data transformations') 
features <- select(dread, 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151') 
w <- select(dread, 'treatment') 
o <- select(dread, 'V152') 
c <- select(dread, 'cost') 
d2d <- select(dread, 'd2d') 

X <- matrix(as.vector(t(features)), , num_features, byrow = TRUE) 
W<- matrix(as.vector(t(w)), , 1) 
Y<- matrix(as.vector(t(o)), , 1) 
C<- matrix(as.vector(t(c)), , 1) 
D2D<- matrix(as.vector(t(d2d)), , 1) 

print('fitting causal random forest for R') 
tauO.forest <- causal_forest(X, Y, W, num.trees=num_trees, alpha=p_alpha, min.node.size=p_min_node_size, sample.fraction=p_sample_fraction) #tune.parameters=TRUE) # 
#tauO.forest$tuning.output 

print('fitting causal random forest for C') 
tauC.forest <- causal_forest(X, C, W, num.trees=num_trees, alpha=p_alpha, min.node.size=p_min_node_size, sample.fraction=p_sample_fraction) #tune.parameters=TRUE) # 
#tauC.forest$tuning.output 

print('fitting causal random forest for D2D') 
tauD.forest <- causal_forest(X, D2D, W, num.trees=num_trees, alpha=p_alpha, min.node.size=p_min_node_size, sample.fraction=p_sample_fraction) #tune.parameters=TRUE) # 
#tauC.forest$tuning.output 

print('reading test data from csv, data transformations') 
dread <- read.csv(file=paste('causal_forest_r_data_', prefix,'_test.csv', sep='')) 

features <- select(dread, 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151') 
w <- select(dread, 'treatment') 
o <- select(dread, 'V152') 
c <- select(dread, 'cost') 
d2d <- select(dread, 'd2d') 

Xtest <- matrix(as.vector(t(features)), , num_features, byrow = TRUE) 

print('performing prediction') 
tauO.hat <- predict(tauO.forest, Xtest) 
tauC.hat <- predict(tauC.forest, Xtest) 
tauD.hat <- predict(tauD.forest, Xtest) 

print('predicting and writing to csv, order') 
tdfO <- as.data.frame(t(tauO.hat)) 
write.csv(tdfO, file = paste('results/causal_forest_grf_test_set_results_O_',prefix, '_numtrees', toString(num_trees), '_alpha', toString(p_alpha), '_min_node_size', toString(p_min_node_size), '_sample_fraction', toString(p_sample_fraction),'_global_iter_',toString(global_iter),'.csv', sep='')) 

print('predicting and writing to csv, cost') 
tdfC <- as.data.frame(t(tauC.hat)) 
write.csv(tdfC, file = paste('results/causal_forest_grf_test_set_results_C_',prefix, '_numtrees', toString(num_trees), '_alpha', toString(p_alpha), '_min_node_size', toString(p_min_node_size), '_sample_fraction', toString(p_sample_fraction),'_global_iter_',toString(global_iter),'.csv', sep='')) 

print('predicting and writing to csv, d2d') 
tdfD <- as.data.frame(t(tauD.hat)) 
write.csv(tdfD, file = paste('results/causal_forest_grf_test_set_results_D_',prefix, '_numtrees', toString(num_trees), '_alpha', toString(p_alpha), '_min_node_size', toString(p_min_node_size), '_sample_fraction', toString(p_sample_fraction),'_global_iter_',toString(global_iter),'.csv', sep='')) 
