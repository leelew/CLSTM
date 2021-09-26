import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import validation
import tensorflow as tf
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor

from CausalLSTM import data
from CausalLSTM.data import get_FLX_inputs, make_train_test_data
from CausalLSTM.model import LSTM, CausalLSTM
from CausalLSTM.tree_causality import CausalTree

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
        validation_split=0.2,

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
    print(label_params[0] + '_QC')

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
    assert np.isnan(np.array(feature)).any() == False, \
        ('Label has NaN value!')

    # make train and test dataset
    train_x, train_y, test_x, test_y, train_mean, train_std, normalized_test_x = make_train_test_data(
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
    ct = CausalTree(
        num_features=len(feature_params),
        name_features=feature_params,
        corr_thresold=corr_thresold,
        mic_thresold=mic_thresold,
        flag=flag,
        depth=depth
    )
    children, child_input_idx, child_state_idx = ct(np.array(feature))
    print(children)
    print(child_input_idx)
    print(child_state_idx)
    print('...done...\n')

    """
    # --------------------------------------------------------------------------
    # 4. Training and inference
    # --------------------------------------------------------------------------
    print('\033[1;31m%s\033[0m' % 'start training!\n')

    
    print('training RF')
    model = RandomForestRegressor()
    model.fit(train_x.reshape(-1, N_t*N_f), train_y.reshape(-1, 1))
    y_pred_rf = model.predict(test_x.reshape(-1, N_t*N_f))
    print('r2 of test dataset is {} of RF'.format(
        r2_score(np.squeeze(y_pred_rf), np.squeeze(test_y))))
    print('...done...\n')
    
    print('training LSTM')
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

    model = LSTM(
        num_hiddens,
        batch_size
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=['mse']
    )

   
    history_lstm = model.fit(
        train_x,
        np.squeeze(train_y),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[checkpoint, lr]
    )
    y_pred_lstm = model.predict(
        test_x,
        batch_size=batch_size
    )

    print('r2 of test dataset is {} of LSTM'.format(
        r2_score(np.squeeze(test_y), np.squeeze(y_pred_lstm))))
    print('...done...\n')
    
    print('training CausalLSTM')
    model = CausalLSTM(
        num_nodes=len(children),
        num_hiddens=num_hiddens,
        children=children,
        child_input_idx=child_input_idx,
        child_state_idx=child_state_idx,
        input_len=len_input,
        batch_size=batch_size
    )

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

    y_pred_clstm = model.predict(
        test_x,
        batch_size=batch_size
    )

    print('r2 of test dataset is {} of Causal LSTM'.format(
        r2_score(np.squeeze(test_y), np.squeeze(y_pred_clstm))))
    print('...done...\n')

    # --------------------------------------------------------------------------
    # 5. Saving
    # --------------------------------------------------------------------------
    # 5.1. save basic info of inputs
    basic_info = feature.describe().transpose()
    basic_info.to_csv(path_output+'info/'+site_name+'_info.csv')

    # 5.2. saving loss during training.
    loss = np.concatenate(
        (history_lstm.history['loss'],
         history_lstm.history['val_loss'],
         history_clstm.history['loss'],
         history_clstm.history['val_loss']),
        axis=-1)
    np.save(path_output + 'loss/' + site_name + '_loss.npy', loss)

    # 5.3. saving forecast and observation.
    def renormalized(inputs):
        return inputs*train_std[-1]+train_mean[-1]

    y_pred_lstm = np.squeeze(renormalized(y_pred_lstm))[:, np.newaxis]
    y_pred_clstm = np.squeeze(renormalized(y_pred_clstm))[:, np.newaxis]
    y_pred_rf = np.squeeze(renormalized(y_pred_rf))[:, np.newaxis]
    y_test = np.squeeze(renormalized(test_y))[:, np.newaxis]

    out = np.concatenate(
        (y_pred_rf,
         y_pred_lstm,
         y_pred_clstm,
         y_test),
        axis=-1)
    np.save(path_output + 'output/'+site_name+'_out.npy', out)
    
    np.save(path_output + 'output/'+site_name+'_feature.npy', normalized_test_x)

    print('...done...\n')

    print('\033[1;31m%s\033[0m' % 'Driver is Lu Li, enjoy your travel!')
    """

if __name__ == '__main__':
    pass
