import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr

from v1.main import main

np.random.seed(1)
tf.compat.v1.set_random_seed(13)


import glob
import os
count=0
l = glob.glob('/tera03/lilu/data/FLX2015_DD/' + '*' + 'csv', recursive=True)
print(l)
for i in l:
    #print(i)
    name = i.split('/')[-1]
    site_name = name.split('_')[1]
    df = pd.read_csv('/tera03/lilu/data/FLX2015_DD/site_summary.csv')
    lat = df['latitude'].values
    lon = df['longitude'].values
    print(site_name)
    #if not os.path.exists('/hard/lilu/clstmcases/7DD_depth-4/' + 'loss/'+site_name+'_loss.npy'):
    #print(site_name)
    if site_name in df['site_name'].values:
        print('1')
        main(site_name=site_name,
            path_input=i,
            path_output='/tera03/lilu/work/CLSTM/clstm-random1/',
            feature_params=['TA_F', 
                        'SW_IN_F', 'LW_IN_F', 
                        'VPD_F', 
                        'PA_F', 'P_F',
                        'WS_F', 'CO2_F_MDS', 
                        'TS_F_MDS_1',
                        'G_F_MDS', 'LE_CORR', 'H_CORR', 
                        'SWC_F_MDS_1'],
            label_params=['SWC_F_MDS_1'],
            # causality params
            corr_thresold=0.5,
            mic_thresold=0.5,
            flag=[1, 0, 0],
            depth=2,

            # model params
            len_input=10,
            len_output=1,
            window_size=7,

            num_hiddens=16,
            batch_size=50,
            epochs=50,
            validation_split=0.2,)

        count+=1
        #pgm.render()
        #pgm.savefig('cs-'+site_name+'.pdf')

