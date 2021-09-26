import tensorflow as tf
import numpy as np
import pandas as pd

print('tensorflow version is {}'.format(tf.__version__))


def get_FLX_quality(inputs, threshold, resolution='DD'):
    """quality of FLUXNET2015 site data.

    Args:
        inputs ([type]): 
            time series of site data.
        threshold ([type]): 
            threshold of length of data. default set 1000 for daily data.
        resolution ([type]):
            time resolution of site data. default set as 'DD'. could be 'HH'.

    Returns:
        quality [type]: 
            quality number of site data. 0 for bad site and 1 for good site.
    """
    num_nan = np.sum(np.isnan(np.array(inputs)))
    length = len(inputs)

    # get threshold for specific time resolution
    if resolution == 'DD':
        threshold = threshold
    elif resolution == 'HH':
        threshold = threshold*48
    else:
        raise ValueError('Must daily or half-hour data.')

    if length < threshold:  # control length of inputs.
        quality = 0
    else:
        if num_nan > 0.1*length:  # control length of NaN value.
            quality = 0
        else:
            quality = 1

    return quality


def get_FLX_inputs(path,
             feature_params=[],
             label_params=[],
             qc_params=[],
             resolution='DD'):
    try:
        # read feature, label and quality flag.
        feature = pd.read_csv(path)[feature_params]
        print(feature.head())
        label = pd.read_csv(path)[label_params]
        qc = pd.read_csv(path)[qc_params]

        # turn -9999 and flag < 1 to NaN.
        feature[feature == -9999.000] = np.nan
        label[label == -9999.000] = np.nan
        label[qc < 1] = np.nan

        # Notes: FLX2015 data always have long NaN array at beginning
        #        of soil moisture. Therefore must remove these long NaN
        #        for specific FLX case.
        gap_idx = np.where(~np.isnan(label))[0]  # label

        label = label[gap_idx[0]:gap_idx[-1]]
        feature = feature[gap_idx[0]:gap_idx[-1]]

        for i in feature.columns:  # feature
            gap_idx = np.where(~np.isnan(feature[i]))[0]

            label = label[gap_idx[0]:gap_idx[-1]]
            feature = feature[gap_idx[0]:gap_idx[-1]]

        # get quality for each FLX site.
        quality = get_FLX_quality(label, threshold=1000, resolution=resolution)

        if quality == 1:
            # interpolate output if quality is 1.
            feature = feature.interpolate(method='linear')
            label = label.interpolate(method='linear')
        else:
            quality = 0
            feature = None
            label = None
    except:
        quality = 0
        feature = None
        label = None

    return feature, label, quality


def make_input_data(inputs, outputs,
                    len_input, len_output, window_size):
    """Generate inputs and outputs for LSTM."""
    # caculate the last time point to generate batch
    end_idx = inputs.shape[0] - len_input - len_output - window_size
    # generate index of batch start point in order
    batch_start_idx = range(end_idx)
    # get batch_size
    batch_size = len(batch_start_idx)
    # generate inputs
    input_batch_idx = [
        (range(i, i + len_input)) for i in batch_start_idx]
    inputs = np.take(inputs, input_batch_idx, axis=0). \
        reshape(batch_size, len_input,
                inputs.shape[1])
    # generate outputs
    output_batch_idx = [
        (range(i + len_input + window_size, i + len_input + window_size +
               len_output)) for i in batch_start_idx]
    outputs = np.take(outputs, output_batch_idx, axis=0). \
        reshape(batch_size,  len_output,
                outputs.shape[1])
    return inputs, outputs


def make_train_test_data(
    df,
    len_input,
    len_output,
    window_size
):
    n = len(df)
    train_df = df[0:int(n*0.8)]
    test_df = df[int(n*0.8):]
    normalized_test_df = test_df

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    train_x, train_y = make_input_data(
        train_df.values[:, :],
        train_df.values[:, -1][:, np.newaxis],
        len_input,
        len_output,
        window_size
    )
    test_x, test_y = make_input_data(
        test_df.values[:, :],
        test_df.values[:, -1][:, np.newaxis],
        len_input,
        len_output,
        window_size)

    # select proper data length to train model
    train_len = train_x.shape[0]
    train_len = 250*(train_len//250)  # (1-validate_ratio)*batch_size

    train_x = train_x[:train_len]
    train_y = train_y[:train_len]

    test_len = test_x.shape[0]
    test_len = 50*(test_len//50)
    test_x = test_x[:test_len]
    test_y = test_y[:test_len]

    normalized_test_x = normalized_test_df[:test_len].values
    
    return train_x, train_y, test_x, test_y, train_mean, train_std, normalized_test_x