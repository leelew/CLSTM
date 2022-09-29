import os
import sys

import daft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy import stats
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import validation
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

sys.path.append('../')
from v1.data import get_FLX_inputs, make_train_test_data
from v1.model import LSTM, CausalLSTM
from v1.tree_causality import CausalTree

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
        len_input=30,  # 10,
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
    print('\033[1;31m%s\033[0m' % '[CLSTM] Read and Processing input data')
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
    train_x, train_y, test_x, test_y, train_mean, train_std, normalized_test_x = \
        make_train_test_data(feature, len_input, len_output, window_size)
    _, N_t, N_f = train_x.shape

    print('the shape of train dataset is {}'.format(train_x.shape))
    print('the shape of test dataset is {}'.format(test_x.shape))

    # --------------------------------------------------------------------------
    # 3. Making causality tree.
    # --------------------------------------------------------------------------
    # calculate causal tree
    print('\033[1;31m%s\033[0m' % '[CLSTM] making causality tree')
    ct = CausalTree(
        num_features=len(feature_params),
        name_features=feature_params,
        corr_thresold=corr_thresold,
        mic_thresold=mic_thresold,
        flag=flag,
        depth=depth
    )
    tree, children, child_input_idx, child_state_idx, adj = ct(
        np.array(feature))
    print(children)
    print(child_input_idx)
    print(child_state_idx)
    print('...done...\n')

    # --------------------------------------------------------------------------
    # 4. Training and inference
    # --------------------------------------------------------------------------
    print('\033[1;31m%s\033[0m' % '[CLSTM] start training!\n')

    print('[CLSTM] training LSTM')
    checkpoint = ModelCheckpoint(
        filepath='./log/',
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

    inputs = tf.keras.layers.Input(shape=(len_input, train_x.shape[-1]))
    outputs = LSTM(num_hiddens, batch_size)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=['mse'])

    history_lstm = model.fit(
        train_x,
        np.squeeze(train_y),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[checkpoint, lr]
    )
    y_pred_lstm = model.predict(test_x, batch_size=batch_size)

    print('r2 of test dataset is {} of LSTM'.format(
        r2_score(np.squeeze(test_y), np.squeeze(y_pred_lstm))))

    print('[CLSTM] training CausalLSTM')

    inputs = tf.keras.layers.Input(shape=(len_input, N_f))
    outputs = CausalLSTM(
        num_nodes=len(children),
        num_hiddens=num_hiddens,
        children=children,
        child_input_idx=child_input_idx,
        child_state_idx=child_state_idx,
        input_len=len_input,
        batch_size=batch_size)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=['mse'])
    history_clstm = model.fit(
        train_x,
        np.squeeze(train_y),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[checkpoint, lr]
    )

    y_pred_clstm = model.predict(test_x, batch_size=batch_size)

    print('r2 of test dataset is {} of Causal LSTM'.format(
        r2_score(np.squeeze(test_y), np.squeeze(y_pred_clstm))))

    # --------------------------------------------------------------------------
    # 5. Saving
    # --------------------------------------------------------------------------
    print('\033[1;31m%s\033[0m' % '[CLSTM] Inference and saving!\n')

    # 5.1. save basic info of inputs
    basic_info = feature.describe().transpose()
    basic_info.to_csv(path_output+'info/'+site_name+'_info.csv')

    # 5.2. saving loss during training.
    loss = np.concatenate(
        (history_lstm.history['loss'],
         history_lstm.history['val_loss'],
         history_clstm.history['loss'],
         history_clstm.history['val_loss']), axis=-1)
    np.save(path_output + 'loss/' + site_name + '_loss.npy', loss)

    # 5.3. saving forecast and observation.
    def renormalized(inputs):
        return inputs*train_std[-1]+train_mean[-1]

    y_pred_lstm = np.squeeze(renormalized(y_pred_lstm))[:, np.newaxis]
    y_pred_clstm = np.squeeze(renormalized(y_pred_clstm))[:, np.newaxis]
    y_test = np.squeeze(renormalized(test_y))[:, np.newaxis]

    out = np.concatenate((y_pred_lstm, y_pred_clstm, y_test), axis=-1)
    np.save(path_output + 'output/'+site_name+'_out.npy', out)
    np.save(path_output + 'output/'+site_name +
            '_feature.npy', normalized_test_x)
    np.save(path_output + 'output/'+site_name+'_adj.npy', adj)

    # 5.4. draw tree causality. see Supplement Material
    def tree2PGM(trees):
        """draw tree causality"""
        color = ['red', 'blue', 'green', 'pink', 'yellow', 'gray']

        pgm = daft.PGM()
        n = daft.Node("sm", "SM", 1, 1)
        pgm.add_node(n)

        b = trees['2'][0]
        for i, num in enumerate(b):
            pgm.add_node('2-'+str(i+1), str(num+1), i+1,
                         2, plot_params={"ec": color[i]})
            pgm.add_edge('2-'+str(i+1), 'sm')

        count = 0
        c = trees['3']
        for j, value in enumerate(c):
            for k in value:
                if k != 12:
                    pgm.add_node('3-'+str(count), str(k+1),
                                 count+1, 3, plot_params={"ec": color[j]})
                    pgm.add_edge('3-'+str(count), '2-'+str(j+1))
                    count += 1
        return pgm
    pgm = tree2PGM(child_input_idx)

    print('\033[1;31m%s\033[0m' % 'Driver is Lu Li, enjoy your travel!')


if __name__ == '__main__':
    l = glob.glob('/hard/lilu/FLX2015_DD/' + '*' + 'csv', recursive=True)
    for i in l:
        name = i.split('/')[-1]
        site_name = name.split('_')[1]
        print(site_name)
        main(site_name=site_name,
             path_input=i,
             path_output='/hard/lilu/clstmcases/',
             feature_params=['TA_F',
                             'SW_IN_F', 'LW_IN_F',
                             'PA_F',
                             'P_F',
                             'SWC_F_MDS_1'],
             label_params=['SWC_F_MDS_1'],

             # causality params
             cond_ind_test='parcorr',
             max_tau=7,
             sig_thres=0.01,
             var_names=['TA', 'SW', 'LW', 'PA', 'P', 'SM'],
             depth=2,
             num_features=6,

             # model params
             len_input=10,
             len_output=1,
             window_size=7,

             num_hiddens=16,
             batch_size=50,
             epochs=50,
             validation_split=0.2,
             ensemble_epochs=1,
             )
