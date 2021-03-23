import tensorflow as tf
from . import common


class Masksembles2D(tf.keras.layers.Layer):
    """
    :class:`Masksembles2D` is high-level class that implements Masksembles approach
    for 2-dimensional inputs (similar to :class:`tensorflow.keras.layers.SpatialDropout1D`).

    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, H, W, C)
        * Output: (N, H, W, C) (same shape as input)

    Examples:

    >>> m = Masksembles2D(4, 2.0)
    >>> inputs = tf.ones([4, 28, 28, 16])
    >>> output = m(inputs)

    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, n: int, scale: float):
        super(Masksembles2D, self).__init__()

        self.n = n
        self.scale = scale

    def build(self, input_shape):
        channels = input_shape[-1]
        masks = common.generation_wrapper(channels, self.n, self.scale)
        self.masks = self.add_weight("masks",
                                     shape=masks.shape,
                                     trainable=False,
                                     dtype="float32")
        self.masks.assign(masks)

    def call(self, inputs, training=False):
        # inputs : [N, H, W, C]
        # masks : [M, C]
        x = tf.stack(tf.split(inputs, self.n))
        # x : [M, N // M, H, W, C]
        # masks : [M, 1, 1, 1, C]
        x = x * self.masks[:, tf.newaxis, tf.newaxis, tf.newaxis]
        x = tf.concat(tf.split(x, self.n), axis=1)
        return tf.squeeze(x, axis=0)


class Masksembles1D(tf.keras.layers.Layer):
    """
    :class:`Masksembles1D` is high-level class that implements Masksembles approach
    for 1-dimensional inputs (similar to :class:`tensorflow.keras.layers.Dropout`).

    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C)
        * Output: (N, C) (same shape as input)

    Examples:

    >>> m = Masksembles1D(4, 2.0)
    >>> inputs = tf.ones([4, 16])
    >>> output = m(inputs)


    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, n: int, scale: float):
        super(Masksembles1D, self).__init__()

        self.n = n
        self.scale = scale

    def build(self, input_shape):
        channels = input_shape[-1]
        masks = common.generation_wrapper(channels, self.n, self.scale)
        self.masks = self.add_weight("masks",
                                     shape=masks.shape,
                                     trainable=False,
                                     dtype="float32")
        self.masks.assign(masks)

    def call(self, inputs, training=False):
        x = tf.stack(tf.split(inputs, self.n))
        x = x * self.masks[:, tf.newaxis]
        x = tf.concat(tf.split(x, self.n), axis=1)
        return tf.squeeze(x, axis=0)
