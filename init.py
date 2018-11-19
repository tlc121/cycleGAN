import numpy as np
import tensorflow as tf
from skimage.transform import resize
import os
from matplotlib.image import imread
from Model import generator, classification, classification_brown

#normalize all image
def normalize(img):
    img = resize(img, output_shape=[256, 256, 3])
    mini = np.min(img)
    maxi = np.max(img)
    img = (img - mini) * 1.0 / (maxi - mini)
    return img

#load data
def data_load(num_images):
    anime_path = './train_anime/'
    real_path = './train_real/'
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

    # DLoss_A_Fake = tf.reduce_mean(tf.square(classification(genA_B)))
    # DLoss_B_Fake = tf.reduce_mean(tf.square(classification(genB_A)))
    # DLoss_A_True = tf.reduce_mean(tf.squared_difference(classification(inputA), 1))
    # DLoss_B_True = tf.reduce_mean(tf.squared_difference(classification(inputB), 1))

    DLoss_A_Fake = tf.reduce_mean(tf.square(classification_brown(genA_B)))
    DLoss_B_Fake = tf.reduce_mean(tf.square(classification_brown(genB_A)))
    DLoss_A_True = tf.reduce_mean(tf.squared_difference(classification_brown(inputA), 1))
    DLoss_B_True = tf.reduce_mean(tf.squared_difference(classification_brown(inputB), 1))

    # Loss Function for Discriminator
    DLoss_A = (DLoss_A_Fake + DLoss_A_True) / 2.0
    DLoss_B = (DLoss_B_Fake + DLoss_B_True) / 2.0

    # GLoss_A_1 = tf.reduce_mean(tf.squared_difference(classification(genA_B), 1))
    # GLoss_B_1 = tf.reduce_mean(tf.squared_difference(classification(genB_A), 1))
    GLoss_A_1 = tf.reduce_mean(tf.squared_difference(classification_brown(genA_B), 1))
    GLoss_B_1 = tf.reduce_mean(tf.squared_difference(classification_brown(genB_A), 1))
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
            with tf.device('/GPU:0'):
                print ('Step' + str(j), 'GLoss_A:', sess.run(GLoss_A, feed_dict={inputA: batch_real, inputB: batch_anime}))
                print ('Step' + str(j), 'GLoss_B:', sess.run(GLoss_B, feed_dict={inputA: batch_real, inputB: batch_anime}))
                print ('Step' + str(j), 'DLoss_A:', sess.run(DLoss_A, feed_dict={inputA: batch_real, inputB: batch_anime}))
                print ('Step' + str(j), 'DLoss_B:', sess.run(DLoss_B, feed_dict={inputA: batch_real, inputB: batch_anime}))
                sess.run(total_loss, feed_dict={inputA: batch_real, inputB: batch_anime})
        if (i + 1) % 10 == 0:
            saver.save(sess, model_path + 'model' + str(i+1))











