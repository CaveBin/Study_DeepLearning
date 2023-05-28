import tensorflow as tf

class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        # 创建一个形状为(input_size, output_size)的矩阵W, 并将其随机初始化
        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        # 创建一个形状为(output_size, )的零向量 b
        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    # 前向传播
    def __call__(self, inputs):
        return self.activation(tf.matmul(input, self.W) + self.b)

    # 获取该层权重的便捷方法
    @property
    def weights(self):
        return [self.W, self.b]