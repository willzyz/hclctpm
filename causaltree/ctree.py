import numpy as np
import pandas as pd

from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.metrics import plot_gain

from sklearn.model_selection import train_test_split
from DataProcFunctions import * 

df, x_names = make_uplift_classification() 

df.head()


# Look at the conversion rate and sample size in each group
df.pivot_table(values='conversion',
               index='treatment_group_key',
               aggfunc=[np.mean, np.size],
               margins=True)

# Split data to training and testing samples for model validation (next section)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=111)

# load the training data for RxGy 
D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromCsvP3('data/rxgy_training_data_v4_samp0.3') 
def text_map_w(w_i):
    if w_i == 0:
        val = 'control'
    else:
        val = 'treatment1'
    return val
D = D[0:3200]
w = w[0:3200]
o = o[0:3200]

wtext = [text_map_w(w_i) for w_i in w]
data = {'treatment_group_key':wtext, 'trips':o}
dframe = pd.DataFrame(data)

uplift_model = UpliftRandomForestClassifier(control_name='control')

import ipdb; ipdb.set_trace() 

d = df_train[x_names].values
uplift_model.fit(d, treatment=df_train['treatment_group_key'].values, y=df_train['conversion'].values) 

import ipdb; ipdb.set_trace() 

uplift_model.fit(D, treatment=dframe['treatment_group_key'].values, y=df_train['conversion'].values) #dframe['trips'].values) 

#treatment=dframe['treatment_group_key'].values, y=dframe['trips'].values)
#uplift_model.fit(d, treatment=dframe['treatment_group_key'].values, y=dframe['trips'].values)

import ipdb; ipdb.set_trace() 

y_pred = uplift_model.predict(df_test[x_names].values)

# Put the predictions to a DataFrame for a neater presentation
result = pd.DataFrame(y_pred,
                      columns=uplift_model.classes_)

