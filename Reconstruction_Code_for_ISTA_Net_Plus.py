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
import glob
from time import time
from PIL import Image
import math


CS_ratio = 25    # 1, 4, 10, 25, 30,, 40, 50

cpkt_model_number = 200


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
PhaseNumber = 9
nrtrain = 88912
learning_rate = 0.0001
EpochNum = 200


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



Test_Img = './Test_Image'


def imread_CS_py(imgName):
    block_size = 33
    Iorg = np.array(Image.open(imgName), dtype='float32')
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


filepaths = glob.glob(Test_Img + '/*.tif')



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


optm_all = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost_all)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)
# sess.run(init)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)


model_dir = 'Phase_%d_ratio_0_%d_ISTA_Net_plus_Model' % (PhaseNumber, CS_ratio)
saver.restore(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, cpkt_model_number))

output_file_name = "PSNR_Results_%s.txt" % (model_dir)
output_file = open(output_file_name, 'a')

for img_no in range(ImgNum):

    imgName = filepaths[img_no]

    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
    Icol = img2col_py(Ipad, 33).transpose()/255.0
    print(Ipad.shape)
    Img_input = np.dot(Icol, Phi_input)
    Img_output = Icol

    start = time()
    Prediction_value = sess.run(Prediction[-1], feed_dict={X_input: Img_input})

    end = time()
    cost_sym_value = sess.run(cost_sym, feed_dict={X_input: Img_input, X_output: Img_output})

    X_rec = col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new)

    rec_PSNR = psnr(X_rec * 255, Iorg)

    print("Run time for %s is %.4f, PSNR is %.2f, loss sym is %.4f" % (imgName, (end - start), rec_PSNR, cost_sym_value))

    img_rec_name = "%s_rec_%s_%d_PSNR_%.2f.png" % (imgName, model_dir, cpkt_model_number, rec_PSNR)

    x_im_rec = Image.fromarray(np.clip(X_rec * 255, 0, 255).astype(np.uint8))
    x_im_rec.save(img_rec_name)

    PSNR_All[0, img_no] = rec_PSNR


output_data = "Avg PSNR is %.2f dB, cpkt NO. is %d \n" % (np.mean(PSNR_All), cpkt_model_number)
print(output_data)
output_file.write(output_data)
output_file.close()


sess.close()

print("Reconstruction READY")