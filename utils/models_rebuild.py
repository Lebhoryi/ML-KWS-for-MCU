# coding=utf-8
'''
@ Summary: 获取网络节点中间变量值
@ Update:  

@ file:    models_rebuild.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/5/6 下午6:29
'''

import tensorflow as tf
import math


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    """Calculates common settings needed for all models.

    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second. 默认的两个 + train时候指定的参数个数
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)         #语音样本点数16k=16000
    window_size_samples = int(sample_rate * window_size_ms / 1000)       #计算一帧的样本个数以及帧移的样本个数，这里它们的值是640和640。
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)        #减去一帧DNN (16000-640)/640+1=25 cnn (16000-640)/320+1=49
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
    }

def get_v(variable, lines):
    variable_index = lines.index(variable) + 1
    try:
        v = lines[variable_index].split(',')
        v = list(map(lambda x: float(x), v))
        return tf.Variable(v)
    except ValueError as v:
        return v


def create_model(fingerprint_input, lines, model_settings, variables,
                 label_count, model_architecture, model_size_info):
    """Builds a model of the requested architecture compatible with the settings.
    """
    if model_architecture == 'dnn':
        return model_dnn(variables, fingerprint_input, lines,
                         model_settings, label_count, model_size_info)
    elif model_architecture == 'cnn':
        return model_cnn(variables, fingerprint_input, lines,
                         model_settings, label_count, model_size_info)
    elif model_architecture == 'ds_cnn':
        return create_ds_cnn_model(variables, fingerprint_input, lines,
                         model_settings, label_count, model_size_info)
    else:
        raise Exception('This have not support {} model.'.format(model_architecture))


def model_dnn(variables, fingerprint_input, lines, model_settings,
              label_count, model_size_info):
    """Builds a model with multiple hidden fully-connected layers.
    model_size_info: length of the array defines the number of hidden-layers and
                     each element in the array represent the number of neurons
                     in that layer
    """

    fingerprint_size = model_settings['fingerprint_size']
    label_count = label_count
    num_layers = len(model_size_info)
    layer_dim = [fingerprint_size]  # [255, 128, 128, 128]
    layer_dim.extend(model_size_info)
    flow = fingerprint_input
    tf.summary.histogram('input', flow)

    output = []  # 存放各层的输出变量
    for i in range(num_layers):
        with tf.variable_scope('fc'+str(i+1)):
            W = get_v(variables[2*i],lines)
            W = tf.reshape(W, (layer_dim[i], layer_dim[i+1]))
            b = get_v(variables[2*i+1], lines)
            flow = tf.matmul(flow, W) + b
            output.append(flow)
            flow = tf.nn.relu(flow)

    weights = get_v(variables[6], lines)
    weights = tf.reshape(weights, (layer_dim[-1], label_count))
    bias = get_v(variables[7], lines)
    logits = tf.matmul(flow, weights) + bias

    return output, logits

def model_cnn(variabels, fingerprint_input, lines, model_settings,
              label_count, model_size_info, is_training=False):
    """Builds a model with 2 convolution layers followed by a linear layer and
          a hidden fully-connected layer.
      model_size_info: defines the first and second convolution parameters in
          {number of conv features, conv filter height, width, stride in y,x dir.},
          followed by linear layer size and fully-connected layer size.
    """

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [1, input_time_size, input_frequency_size, 1])
    # tmp = fingerprint_4d.eval()
    first_filter_count = model_size_info[0]
    first_filter_height = model_size_info[1]   #time axis
    first_filter_width = model_size_info[2]    #frequency axis
    first_filter_stride_y = model_size_info[3] #time axis
    first_filter_stride_x = model_size_info[4] #frequency_axis

    second_filter_count = model_size_info[5]
    second_filter_height = model_size_info[6]   #time axis
    second_filter_width = model_size_info[7]    #frequency axis
    second_filter_stride_y = model_size_info[8] #time axis
    second_filter_stride_x = model_size_info[9] #frequency_axis

    linear_layer_size = model_size_info[10]
    fc_size = model_size_info[11]


    # first conv
    # get weight
    first_weights = get_v(variabels[0], lines)
    first_weights = tf.reshape(first_weights,
                               (first_filter_height, first_filter_width, 1, first_filter_count))
    # get bias
    first_bias = get_v(variabels[1], lines)
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
        1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias

    first_conv = tf.layers.batch_normalization(first_conv, training=is_training,
                                               name='bn1')
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_conv_output_width = math.ceil(
        (input_frequency_size - first_filter_width + 1) /
        first_filter_stride_x)
    first_conv_output_height = math.ceil(
        (input_time_size - first_filter_height + 1) /
        first_filter_stride_y)

    # second conv
    # get weight
    second_weights = get_v(variabels[2], lines)
    second_weights = tf.reshape(second_weights,
                                (second_filter_height, second_filter_width, first_filter_count,
                                 second_filter_count))

    # get bias
    second_bias = get_v(variabels[3], lines)
    second_conv = tf.nn.conv2d(first_dropout, second_weights, [
        1, second_filter_stride_y, second_filter_stride_x, 1
    ], 'VALID') + second_bias
    second_conv = tf.layers.batch_normalization(second_conv, training=is_training,
                                                name='bn2')
    second_relu = tf.nn.relu(second_conv)
    if is_training:
        second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    else:
        second_dropout = second_relu
    second_conv_output_width = math.ceil(
        (first_conv_output_width - second_filter_width + 1) /
        second_filter_stride_x)
    second_conv_output_height = math.ceil(
        (first_conv_output_height - second_filter_height + 1) /
        second_filter_stride_y)

    second_conv_element_count = int(
        second_conv_output_width*second_conv_output_height*second_filter_count)
    flattened_second_conv = tf.reshape(second_dropout,
                                       [1, second_conv_element_count])

    # linear layer
    # get weight
    W = get_v(variabels[4], lines)
    W = tf.reshape(W, (second_conv_element_count, linear_layer_size))
    # get bias
    b = get_v(variabels[5], lines)
    flow = tf.matmul(flattened_second_conv, W) + b

    # first fc
    first_fc_output_channels = fc_size
    # get weight
    first_fc_weights = get_v(variabels[6], lines)
    first_fc_weights = tf.reshape(first_fc_weights,
                                  (linear_layer_size, first_fc_output_channels))
    # get bias
    first_fc_bias = get_v(variabels[7], lines)

    first_fc = tf.matmul(flow, first_fc_weights) + first_fc_bias
    first_fc = tf.layers.batch_normalization(first_fc, training=is_training,
                                             name='bn3')
    first_fc_relu = tf.nn.relu(first_fc)

    if is_training:
        final_fc_input = tf.nn.dropout(first_fc_relu, dropout_prob)
    else:
        final_fc_input = first_fc_relu

    # final fc
    # get weight
    final_fc_weights = get_v(variabels[8], lines)
    final_fc_weights = tf.reshape(final_fc_weights,
                                  (first_fc_output_channels, label_count))
    # get bias
    final_fc_bias = get_v(variabels[9], lines)

    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias

    if is_training:
        return final_fc, dropout_prob
    else:
        return first_conv, second_conv, flow, first_fc, final_fc


def create_ds_cnn_model(fingerprint_input, model_settings, model_size_info,
                        is_training):
    """Builds a model with depthwise separable convolutional neural network
    Model definition is based on https://arxiv.org/abs/1704.04861 and
    Tensorflow implementation: https://github.com/Zehaos/MobileNet

    model_size_info: defines number of layers, followed by the DS-Conv layer
      parameters in the order {number of conv features, conv filter height,
      width and stride in y,x dir.} for each of the layers.
    Note that first layer is always regular convolution, but the remaining
      layers are all depthwise separable convolutions.
    """

    def ds_cnn_arg_scope(weight_decay=0):
        """Defines the default ds_cnn argument scope.
        Args:
          weight_decay: The weight decay to use for regularizing the model.
        Returns:
          An `arg_scope` to use for the DS-CNN model.
        """
        with slim.arg_scope(
                [slim.convolution2d, slim.separable_convolution2d],
                weights_initializer=slim.initializers.xavier_initializer(),
                biases_initializer=slim.init_ops.zeros_initializer(),
                weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
            return sc

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  sc,
                                  kernel_size,
                                  stride):
        """ Helper function to build the depth-wise separable convolution layer.
        """

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=stride,
                                                      depth_multiplier=1,
                                                      kernel_size=kernel_size,
                                                      scope=sc+'/dw_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_conv/batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc+'/pw_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_conv/batch_norm')
        return bn


    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    label_count = model_settings['label_count']
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])

    t_dim = input_time_size
    f_dim = input_frequency_size

    #Extract model dimensions from model_size_info
    num_layers = model_size_info[0]
    conv_feat = [None]*num_layers
    conv_kt = [None]*num_layers
    conv_kf = [None]*num_layers
    conv_st = [None]*num_layers
    conv_sf = [None]*num_layers
    i=1
    for layer_no in range(0,num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    scope = 'DS-CNN'
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            biases_initializer=slim.init_ops.zeros_initializer(),
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                decay=0.96,
                                updates_collections=None,
                                activation_fn=tf.nn.relu):
                for layer_no in range(0,num_layers):
                    if layer_no==0:
                        net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no], \
                                                 [conv_kt[layer_no], conv_kf[layer_no]], stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME', scope='conv_1')
                        net = slim.batch_norm(net, scope='conv_1/batch_norm')
                    else:
                        net = _depthwise_separable_conv(net, conv_feat[layer_no], \
                                                        kernel_size = [conv_kt[layer_no],conv_kf[layer_no]], \
                                                        stride = [conv_st[layer_no],conv_sf[layer_no]], sc='conv_ds_'+str(layer_no))
                    t_dim = math.ceil(t_dim/float(conv_st[layer_no]))
                    f_dim = math.ceil(f_dim/float(conv_sf[layer_no]))

                net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')

        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')

    if is_training:
        return logits, dropout_prob
    else:
        return logits
