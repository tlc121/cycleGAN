import numpy as np
import tensorflow as tf
from Layers import conv2d, deconvnet, build_resnet_block

#This is the first part of the Generator Networks
def convnet(input, Activation, relu_factor):
    conv1 = conv2d(name='conv1G', input=input, kernel_size=4, output=64, Activation=Activation, relu_factor=relu_factor,
                   pad='SAME')
    conv2 = conv2d(name='conv2G', input=conv1, kernel_size=4, output=128, Activation=Activation,
                   relu_factor=relu_factor, pad='SAME')
    conv3 = conv2d(name='conv3G', input=conv2, kernel_size=4, output=256, Activation=Activation,
                   relu_factor=relu_factor, pad='SAME')
    return conv3

#Second part of the G network
def transform(input):
    o_r1 = build_resnet_block(inputres=input, dim=256, name="r1")
    o_r2 = build_resnet_block(inputres=o_r1, dim=256, name="r2")
    o_r3 = build_resnet_block(inputres=o_r2, dim=256, name="r3")
    o_r4 = build_resnet_block(inputres=o_r3, dim=256, name="r4")
    o_r5 = build_resnet_block(inputres=o_r4, dim=256, name="r5")
    o_r6 = build_resnet_block(inputres=o_r5, dim=256, name="r6")
    return o_r6

#Final part of G, reshape back to original size
def deconvNet(input):
    deconv1 = deconvnet(name='deconv1', input=input, kernel=4, output=128, Activation=tf.nn.leaky_relu, relu_factor=0.2)
    deconv2 = deconvnet(name='deconv2', input=deconv1, kernel=4, output=64, Activation=tf.nn.leaky_relu,relu_factor=0.2)
    deconv3 = deconvnet(name='deconv3', input=deconv2, kernel=4, output=3, Activation=tf.nn.leaky_relu, relu_factor=0.2)
    return deconv3


#Generator Network
def generator(input):
    encode = convnet(input=input, Activation=tf.nn.leaky_relu, relu_factor=0.2)
    trans = transform(encode)
    decode = deconvNet(trans)
    return decode


#Discriminator Network
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

