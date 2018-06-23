'''
Test Platform: Tensorflow version: 1.2.0

Paper Information:

ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing
Jian Zhang and Bernard Ghanem
IEEE Conference on Computer Vision and Pattern Recognition (CVPR2018), Salt Lake City, USA, Jun. 2018.

Email: jianzhang.tech@gmail.com
2018/5/16
'''


import tensorflow as tf
import scipy.io as sio
import numpy as np
import os


CS_ratio = 25    # 4, 10, 25, 30,, 40, 50


if CS_ratio == 4:
    n_input = 43
elif CS_ratio == 1:
    n_input = 10
elif CS_ratio == 10:
    n_input = 109
elif CS_ratio == 25:
    n_input = 272
elif CS_ratio == 30:
    n_input = 327
elif CS_ratio == 40:
    n_input = 436
elif CS_ratio == 50:
    n_input = 545


n_output = 1089
batch_size = 64
PhaseNumber = 5
nrtrain = 88912
learning_rate = 0.0001
EpochNum = 300


print('Load Data...')

Phi_data_Name = 'phi_0_%d_1089.mat' % CS_ratio
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi'].transpose()

Training_data_Name = 'Training_Data_Img91.mat'
Training_data = sio.loadmat(Training_data_Name)
Training_inputs = Training_data['inputs']
Training_labels = Training_data['labels']


# Computing Initialization Matrix
XX = Training_labels.transpose()
BB = np.dot(Phi_input.transpose(), XX)
BBB = np.dot(BB, BB.transpose())
CCC = np.dot(XX, BB.transpose())
PhiT_ = np.dot(CCC, np.linalg.inv(BBB))
del XX, BB, BBB, Training_data
PhiInv_input = PhiT_.transpose()
PhiTPhi_input = np.dot(Phi_input, Phi_input.transpose())


Phi = tf.constant(Phi_input, dtype=tf.float32)
PhiTPhi = tf.constant(PhiTPhi_input, dtype=tf.float32)
PhiInv = tf.constant(PhiInv_input, dtype=tf.float32)

X_input = tf.placeholder(tf.float32, [None, n_input])
X_output = tf.placeholder(tf.float32, [None, n_output])


X0 = tf.matmul(X_input, PhiInv)

PhiTb = tf.matmul(X_input, tf.transpose(Phi))


def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]


def ista_block(input_layers, input_data, layer_no):
    tau_value = tf.Variable(0.1, dtype=tf.float32)
    lambda_step = tf.Variable(0.1, dtype=tf.float32)
    soft_thr = tf.Variable(0.1, dtype=tf.float32)
    conv_size = 32
    filter_size = 3

    x1_ista = tf.add(input_layers[-1] - tf.scalar_mul(lambda_step, tf.matmul(input_layers[-1], PhiTPhi)), tf.scalar_mul(lambda_step, PhiTb))  # X_k - lambda*A^TAX

    x2_ista = tf.reshape(x1_ista, shape=[-1, 33, 33, 1])

    [Weights0, bias0] = add_con2d_weight_bias([filter_size, filter_size, 1, conv_size], [conv_size], 0)

    [Weights1, bias1] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 1)
    [Weights11, bias11] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 11)

    [Weights2, bias2] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 2)
    [Weights22, bias22] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 22)

    [Weights3, bias3] = add_con2d_weight_bias([filter_size, filter_size, conv_size, 1], [1], 3)

    x3_ista = tf.nn.conv2d(x2_ista, Weights0, strides=[1, 1, 1, 1], padding='SAME')

    x4_ista = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x44_ista = tf.nn.conv2d(x4_ista, Weights11, strides=[1, 1, 1, 1], padding='SAME')

    x5_ista = tf.multiply(tf.sign(x44_ista), tf.nn.relu(tf.abs(x44_ista) - soft_thr))

    x6_ista = tf.nn.relu(tf.nn.conv2d(x5_ista, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x66_ista = tf.nn.conv2d(x6_ista, Weights22, strides=[1, 1, 1, 1], padding='SAME')

    x7_ista = tf.nn.conv2d(x66_ista, Weights3, strides=[1, 1, 1, 1], padding='SAME')

    x7_ista = x7_ista + x2_ista

    x8_ista = tf.reshape(x7_ista, shape=[-1, 1089])

    x3_ista_sym = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x4_ista_sym = tf.nn.conv2d(x3_ista_sym, Weights11, strides=[1, 1, 1, 1], padding='SAME')
    x6_ista_sym = tf.nn.relu(tf.nn.conv2d(x4_ista_sym, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x7_ista_sym = tf.nn.conv2d(x6_ista_sym, Weights22, strides=[1, 1, 1, 1], padding='SAME')

    x11_ista = x7_ista_sym - x3_ista

    return [x8_ista, x11_ista]


def inference_ista(input_tensor, n, X_output, reuse):
    layers = []
    layers_symetric = []
    layers.append(input_tensor)
    for i in range(n):
        with tf.variable_scope('conv_%d' %i, reuse=reuse):
            [conv1, conv1_sym] = ista_block(layers, X_output, i)
            layers.append(conv1)
            layers_symetric.append(conv1_sym)
    return [layers, layers_symetric]


[Prediction, Pre_symetric] = inference_ista(X0, PhaseNumber, X_output, reuse=False)

cost0 = tf.reduce_mean(tf.square(X0 - X_output))


def compute_cost(Prediction, X_output, PhaseNumber):
    cost = tf.reduce_mean(tf.square(Prediction[-1] - X_output))
    cost_sym = 0
    for k in range(PhaseNumber):
        cost_sym += tf.reduce_mean(tf.square(Pre_symetric[k]))

    return [cost, cost_sym]


[cost, cost_sym] = compute_cost(Prediction, X_output, PhaseNumber)


cost_all = cost + 0.01*cost_sym


optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)
sess.run(init)

print("...............................")
print("Phase Number is %d, CS ratio is %d%%" % (PhaseNumber, CS_ratio))
print("...............................\n")

print("Strart Training..")


model_dir = 'Phase_%d_ratio_0_%d_ISTA_Net_plus_Model' % (PhaseNumber, CS_ratio)

output_file_name = "Log_output_%s.txt" % (model_dir)

for epoch_i in range(0, EpochNum+1):
    randidx_all = np.random.permutation(nrtrain)
    for batch_i in range(nrtrain // batch_size):
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

        batch_ys = Training_labels[randidx, :]
        batch_xs = np.dot(batch_ys, Phi_input)

        feed_dict = {X_input: batch_xs, X_output: batch_ys}
        sess.run(optm_all, feed_dict=feed_dict)

    output_data = "[%02d/%02d] cost: %.4f, cost_sym: %.4f \n" % (epoch_i, EpochNum, sess.run(cost, feed_dict=feed_dict), sess.run(cost_sym, feed_dict=feed_dict))
    print(output_data)

    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if epoch_i <= 30:
        saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
    else:
        if epoch_i % 20 == 0:
            saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)


print("Training Finished")
sess.close()