import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import ensemble
from sklearn.utils import validation
import tensorflow as tf
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor

from src import data
from src.data import get_FLX_inputs, make_train_test_data
from src.model import CausalLSTM
from src.tree_causality import CausalPrecursors

np.random.seed(1)
tf.compat.v1.set_random_seed(13)


def main(
        # input data params
        site_name='',
        path_input='',
        path_output='',
        feature_params=[],
        label_params=[],

        # causality params
        cond_ind_test='parcorr',
        max_tau=7,
        sig_thres=0.05,
        var_names=['TA','SW','LW','PA','P','WS','TS','SM'],
        depth=2, 
        num_features=8,

        # model params
        len_input=10,
        len_output=1,
        window_size=7,

        num_hiddens=16,
        batch_size=50,
        epochs=50,
        validation_split=0.2,
        ensemble_epochs=2,
):
    # --------------------------------------------------------------------------
    # 1. make output dir.
    # --------------------------------------------------------------------------
    if not os.path.exists(path_output + 'output/'):
        os.mkdir(path_output + 'output/')

    if not os.path.exists(path_output + 'loss/'):
        os.mkdir(path_output + 'loss/')

    if not os.path.exists(path_output + 'info/'):
        os.mkdir(path_output + 'info/')

    # --------------------------------------------------------------------------
    # 2. read and preprocessing FLUXNET2015 dataset.
    # --------------------------------------------------------------------------
    # process data
    print('\033[1;31m%s\033[0m' % 'Read and Processing input data')

    qc_params = []
    qc_params.append(label_params[0]+'_QC')
    feature, label, quality = get_FLX_inputs(
        path=path_input,
        feature_params=feature_params,
        label_params=label_params,
        qc_params=qc_params,
        resolution='DD',
    )

    if quality == 0:
        print('This site cannot be used, careful for your inputs!')
        return

    # assert feature/label have any NaN
    assert np.isnan(np.array(feature)).any() == False, \
        ('Features have NaN value!')
    assert np.isnan(np.array(label)).any() == False, \
        ('Label has NaN value!')

    # make train and test dataset
    train_x, train_y, test_x, test_y, train_mean, train_std, normalized_test_x \
        = make_train_test_data( \
        feature,
        len_input,
        len_output,
        window_size
    )
    _, N_t, N_f = train_x.shape

    print('the shape of train dataset is {}'.format(train_x.shape))
    print('the shape of test dataset is {}'.format(test_x.shape))

    print('...done...\n')
    
    # --------------------------------------------------------------------------
    # 3. Making causality tree.
    # --------------------------------------------------------------------------
    # calculate causal tree
    print('\033[1;31m%s\033[0m' % 'making causality tree')
    
    a = CausalPrecursors(
            site_name=site_name,
            cond_ind_test=cond_ind_test,
            max_tau=max_tau,
            sig_thres=sig_thres,
            var_names=var_names,
            depth=depth, 
            num_features=num_features)(feature)
    print(a.group_node_dict)
    print(a.group_num_child_nodes)
    print(a.group_input_idx)
    print(a.group_child_state_idx)
    print('...done...\n')

    # --------------------------------------------------------------------------
    # 4. Training and inference
    # --------------------------------------------------------------------------
    print('\033[1;31m%s\033[0m' % 'start training!\n')

    print('training Forest Causal LSTM')
    checkpoint = ModelCheckpoint(
        filepath='/Users/lewlee/Desktop/log/',
        monitor='val_loss',
        save_best_only='True',
        save_weights_only='True'
    )

    lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
        verbose=0,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0
    )

    print('training CausalLSTM')

    N_test = test_y.shape[0]
    N_train = train_x.shape[0]
    num_tree = len(a.group_input_idx.keys())
    y_pred_clstm = np.full((N_test, num_tree*ensemble_epochs), np.nan)
    y_train_clstm = np.full((N_train, num_tree*ensemble_epochs), np.nan)

    for j in np.arange(ensemble_epochs):
        for i, timestep in enumerate(a.group_num_child_nodes.keys()):
            num_child_nodes = a.group_num_child_nodes[timestep]
            input_idx = a.group_input_idx[timestep]
            child_state_idx = a.group_child_state_idx[timestep]

            num_nodes = len(num_child_nodes)
                
            model = CausalLSTM(                 
                    num_child_nodes=num_child_nodes,
                    input_idx=input_idx,
                    child_state_idx=child_state_idx,
                    num_nodes=num_nodes,
                    num_hiddens=num_hiddens,
                    input_len=len_input,
                    batch_size=batch_size)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=['mse']
            )

            history_clstm = model.fit(
                train_x,
                np.squeeze(train_y),
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=[checkpoint, lr]
            )
            y_train_clstm[:,num_tree*j+i] = np.squeeze(model.predict(
                train_x,
                batch_size=batch_size
            ))
            y_pred_clstm[:,num_tree*j+i] = np.squeeze(model.predict(
                test_x,
                batch_size=batch_size
            ))

    y_pred_clstm = np.nanmean(y_pred_clstm, axis=-1)
    test_y = np.squeeze(test_y)
    print('r2 of test dataset is {} of Causal LSTM'.format(
        r2_score(np.squeeze(test_y), np.squeeze(y_pred_clstm))))

    # --------------------------------------------------------------------------
    # 5. Saving
    # --------------------------------------------------------------------------
    def renormalized(inputs):
        return inputs*train_std[-1]+train_mean[-1]

    y_pred_clstm = np.squeeze(renormalized(y_pred_clstm))[:, np.newaxis]
    y_test = np.squeeze(renormalized(test_y))[:, np.newaxis]

    out = np.concatenate(
        (y_pred_clstm,
         y_test),
        axis=-1)
    
    np.save(path_output + 'output/'+site_name+'_out.npy', out)

    print('...done...\n')

if __name__ == '__main__':
    pass
