import tensorflow as tf

class SplitLayer(tf.keras.layers.Layer):
    def __init__(self, num_splits, **kwargs):
        super(SplitLayer, self).__init__(**kwargs)
        self.num_splits = num_splits

    def call(self, inputs):
        return tf.split(inputs, num_or_size_splits=self.num_splits, axis=1)
