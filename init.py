import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
import os
from matplotlib.image import imread


def normalize(img):
    img = resize(img, output_shape=[256, 256, 3])
    mini = np.min(img)
    maxi = np.max(img)
    img = (img - mini) * 1.0 / (maxi - mini)
    return img


def data_load(num_images):
    anime_path = './train_anime/train_anime/'
    real_path = './train_real/train_real/'
    anime = np.zeros(shape=[num_images, 256, 256, 3])
    real = np.zeros(shape=[num_images, 256, 256, 3])
    count = 0
    for data in os.listdir(anime_path):
        if count >= num_images:
            break
        if data.endswith('DS_Store'):
            continue
        temp_img = normalize(imread(anime_path + data))
        anime[count] = temp_img
        count += 1

    count = 0
    for data in os.listdir(real_path):
        if count >= num_images:
            break
        if data.endswith('DS_Store'):
            continue
        temp_img = normalize(imread(real_path + data))
        real[count] = temp_img
        count += 1

    return anime, real


# reshape the conv_layer into original image size
def deconvnet(name, input, output, kernel, Activation, relu_factor):
    with tf.name_scope(name):
        conv = tf.layers.conv2d_transpose(inputs=input, filters=output, kernel_size=kernel, strides=[2, 2],
                                          padding='SAME')
        if Activation is None:
            return conv
        return Activation(conv, alpha=relu_factor)


def conv2d(name, input, kernel_size, output, pad, strides=2, Activation=None, relu_factor=0.0):
    with tf.variable_scope(name):
        input_size = input.get_shape()[-1].value
        shape = [kernel_size, kernel_size, input_size, output]
        weights = tf.get_variable(name=name + 'w', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        cons = tf.constant(0.0, shape=[output])
        bias = tf.Variable(cons, name=name + 'b')
        conv = tf.nn.conv2d(input, weights, strides=[1, strides, strides, 1], padding=pad)
        res = tf.nn.bias_add(conv, bias)
        if Activation is None:
            return res
        return Activation(res, alpha=relu_factor)


def build_resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(name='res1', input=out_res, output=dim, kernel_size=3, strides=1, Activation=tf.nn.leaky_relu,
                         relu_factor=0.2, pad='VALID')
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(name='res2', input=out_res, output=dim, kernel_size=3, strides=1, Activation=None, pad='VALID')

        return tf.nn.relu(out_res + inputres)


def convnet(input, Activation, relu_factor):
    conv1 = conv2d(name='conv1G', input=input, kernel_size=4, output=64, Activation=Activation, relu_factor=relu_factor,
                   pad='SAME')
    conv2 = conv2d(name='conv2G', input=conv1, kernel_size=4, output=128, Activation=Activation,
                   relu_factor=relu_factor, pad='SAME')
    conv3 = conv2d(name='conv3G', input=conv2, kernel_size=4, output=256, Activation=Activation,
                   relu_factor=relu_factor, pad='SAME')

    return conv3


def transform(input):
    o_r1 = build_resnet_block(inputres=input, dim=256, name="r1")
    o_r2 = build_resnet_block(inputres=o_r1, dim=256, name="r2")
    o_r3 = build_resnet_block(inputres=o_r2, dim=256, name="r3")
    o_r4 = build_resnet_block(inputres=o_r3, dim=256, name="r4")
    o_r5 = build_resnet_block(inputres=o_r4, dim=256, name="r5")
    o_r6 = build_resnet_block(inputres=o_r5, dim=256, name="r6")
    return o_r6


def deconvNet(input):
    deconv1 = deconvnet(name='deconv1', input=input, kernel=4, output=128, Activation=tf.nn.leaky_relu, relu_factor=0.2)
    deconv2 = deconvnet(name='deconv2', input=deconv1, kernel=4, output=64, Activation=tf.nn.leaky_relu,
                        relu_factor=0.2)
    deconv3 = deconvnet(name='deconv3', input=deconv2, kernel=4, output=3, Activation=tf.nn.leaky_relu, relu_factor=0.2)
    return deconv3


def generator(input):
    encode = convnet(input=input, Activation=tf.nn.leaky_relu, relu_factor=0.2)
    trans = transform(encode)
    decode = deconvNet(trans)
    return decode


def classification(input):
    # full conv layers
    conv1 = conv2d(name='conv1D', input=input, kernel_size=4, output=64, pad='SAME', Activation=tf.nn.leaky_relu,
                   relu_factor=0.2)
    conv2 = conv2d(name='conv2D', input=conv1, kernel_size=4, output=128, pad='SAME', Activation=tf.nn.leaky_relu,
                   relu_factor=0.2)
    conv3 = conv2d(name='conv3D', input=conv2, kernel_size=4, output=256, pad='SAME', Activation=tf.nn.leaky_relu,
                   relu_factor=0.2)
    conv4 = conv2d(name='conv4D', input=conv3, kernel_size=4, output=512, pad='SAME', Activation=tf.nn.leaky_relu,
                   relu_factor=0.2)
    conv5 = conv2d(name='conv5D', input=conv4, kernel_size=4, output=1, strides=1, pad='SAME', Activation=None)
    return conv5

if __name__ == '__main__':
    Batch_Size = 1
    cyc_loss_Factor = 10
    lr = 0.00001
    num_images = 100
    num_epochs = 30
    model_path = './model/'
    anime, real = data_load(num_images)
    batch_num = int(num_images / Batch_Size)

    batch_real = np.zeros(shape=[Batch_Size, 256, 256, 3])
    batch_anime = np.zeros(shape=[Batch_Size, 256, 256, 3])

    sess = tf.Session()
    inputA = tf.placeholder(tf.float32, [None, 256, 256, 3])
    inputB = tf.placeholder(tf.float32, [None, 256, 256, 3])

    genA_B = generator(input=inputA)
    genB_A = generator(input=inputB)
    cycleA = generator(input=genA_B)
    cycleB = generator(input=genB_A)

    DLoss_A_Fake = tf.reduce_mean(tf.square(classification(genA_B)))
    DLoss_B_Fake = tf.reduce_mean(tf.square(classification(genB_A)))
    DLoss_A_True = tf.reduce_mean(tf.squared_difference(classification(inputA), 1))
    DLoss_B_True = tf.reduce_mean(tf.squared_difference(classification(inputB), 1))

    # Loss Function for Discriminator
    DLoss_A = (DLoss_A_Fake + DLoss_A_True) / 2.0
    DLoss_B = (DLoss_B_Fake + DLoss_B_True) / 2.0

    GLoss_A_1 = tf.reduce_mean(tf.squared_difference(classification(genA_B), 1))
    GLoss_B_1 = tf.reduce_mean(tf.squared_difference(classification(genB_A), 1))
    cyc_loss = tf.reduce_mean(tf.abs(cycleA - inputA)) + tf.reduce_mean(tf.abs(cycleB - inputB))

    # Loss for Generator
    GLoss_A = GLoss_A_1 + cyc_loss_Factor * cyc_loss
    GLoss_B = GLoss_B_1 + cyc_loss_Factor * cyc_loss

    optimizer = tf.train.AdamOptimizer(lr, 0.5)
    loss1 = optimizer.minimize(GLoss_A)
    loss2 = optimizer.minimize(GLoss_B)
    loss3 = optimizer.minimize(DLoss_A)
    loss4 = optimizer.minimize(DLoss_B)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i in range(num_epochs):
        for j in range(batch_num):
            start = j * Batch_Size
            end = (j + 1) * Batch_Size
            batch_real = real[start:end, :, :, :]
            batch_anime = anime[start:end, :, :, :]
            total_loss = tf.group(loss1, loss2, loss3, loss4)
            # sess.run(total_loss, feed_dict={inputA: batch_real, inputB: batch_anime})
            with tf.device('/GPU:0'):
                print (
                'Step' + str(j), 'GLoss_A:', sess.run(GLoss_A, feed_dict={inputA: batch_real, inputB: batch_anime}))
                print (
                'Step' + str(j), 'GLoss_B:', sess.run(GLoss_B, feed_dict={inputA: batch_real, inputB: batch_anime}))
                print (
                'Step' + str(j), 'DLoss_A:', sess.run(DLoss_A, feed_dict={inputA: batch_real, inputB: batch_anime}))
                print (
                'Step' + str(j), 'DLoss_B:', sess.run(DLoss_B, feed_dict={inputA: batch_real, inputB: batch_anime}))
                sess.run(total_loss, feed_dict={inputA: batch_real, inputB: batch_anime})
        if (i + 1) % 10 == 0:
            saver.save(sess, model_path + 'model' + str(i))
    with tf.device('/GPU:0'):
        test = sess.run(genA_B, feed_dict={inputA: real[0:1]})
    plt.imshow(test[0])
    plt.show()












