import tensorflow as tf

from utils.resnet_model import conv2d_fixed_padding, batch_norm, block_layer
from utils.resnet_model import _bottleneck_block_v1, _building_block_v1


class ResNet():
    def __init__(self, resnet_size, data_format=None, num_classes=None):
        self.resnet_size = resnet_size
        self.num_filters = 64
        self.kernel_size = 7
        self.conv_stride = 2
        self.first_pool_size = 3
        self.first_pool_stride = 2
        self.block_strides = [1, 2, 2, 2]
        self.num_classes = num_classes
        if resnet_size < 50:
            self.bottleneck = False
            self.final_size = 512
        else:
            self.bottleneck = True
            self.final_size = 2048
            if self.bottleneck:
                self.block_fn = _bottleneck_block_v1
            else:
                self.block_fn = _building_block_v1
        self.block_sizes = self._get_block_sizes(resnet_size)
        if not data_format:
            self.data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    
    def _get_block_sizes(self, resnet_size):
        """Retrieve the size of each block_layer in the ResNet model."""
        choices = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
            200: [3, 24, 36, 3]
        }
        try:
            return choices[resnet_size]
        except KeyError:
            err = ('Could not find layers for selected Resnet size.\n'
                'Size received: {}; sizes allowed: {}.'.format(
                    resnet_size, choices.keys()))
            raise ValueError(err)

    def _preprocess(self, inputs):
        with tf.name_scope('preprocess'):
            mean = tf.constant([123.68, 116.779, 103.939],
                            dtype=tf.float32, shape=[1, 1, 1, 3],
                            name='img_mean')
            return inputs-mean

    def __call__(self, inputs, training, imnet_layer=False, ret_pre_pool=False):
        with tf.variable_scope("resnet_model"):
            inputs = self._preprocess(inputs)
            if self.data_format == 'channels_first':
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                inputs = tf.transpose(inputs, [0, 3, 1, 2])
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=self.num_filters, 
                kernel_size=self.kernel_size,
                strides=self.conv_stride, data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            # using resnet v1
            inputs = batch_norm(inputs, training, self.data_format)
            inputs = tf.nn.relu(inputs)
            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME',
                    data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_max_pool')
            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2**i)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters, 
                    bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_strides[i], training=training,
                    name='block_layer{}'.format(i + 1), 
                    data_format=self.data_format)
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            if ret_pre_pool:
                # Get tensor of pre-average pooling layer
                return inputs
            inputs = tf.reduce_mean(inputs, axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')
            inputs = tf.reshape(inputs, [-1, self.final_size])
            if imnet_layer:
                inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
                inputs = tf.identity(inputs, 'final_dense')
            return inputs