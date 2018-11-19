import tensorflow as tf

#convolutional layers
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

#dilated convolution layers
def dilated_conv2d(name, input, kernel_size, output, rate=1, pad, Activation=None, relu_factor=0.0):
        input_size = input.get_shape()[-1].value
        shape = [kernel_size, kernel_size, input_size, output]
        weights = tf.get_variable(name=name + 'w', shape=shape, initializertf.contrib.layers.xavier_initializer())
        cons = tf.constant(0.0, shape=[output])
        bias = tf.Variable(cons, name=name + 'b')
        dilated_conv = tf.nn.atrous_conv2d(input, weights, rate, padding=pad)
        res = tf.nn.bias_add(dilated_conv, bias)
        if Activation is None:
            return res
        return Activation(res, alpha=relu_factor)

#residual block
def build_resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(name='res1', input=out_res, output=dim, kernel_size=3, strides=1, Activation=tf.nn.leaky_relu,
                         relu_factor=0.2, pad='VALID')
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(name='res2', input=out_res, output=dim, kernel_size=3, strides=1, Activation=None, pad='VALID')
        return tf.nn.relu(out_res + inputres)


# This is the deconvolution layer. Reshape the feature maps into original image size
def deconvnet(name, input, output, kernel, Activation, relu_factor):
    with tf.name_scope(name):
        conv = tf.layers.conv2d_transpose(inputs=input, filters=output, kernel_size=kernel, strides=[2, 2],
                                          padding='SAME')
        if Activation is None:
            return conv
        return Activation(conv, alpha=relu_factor)