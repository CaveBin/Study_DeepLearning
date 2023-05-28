import tensorflow as tf
from tensorflow.python.keras import optimizers

learning_rate = 1e-3

def one_training_step(model, images_batch, labels_batch):
    # 运行前向传播，即在GradientTape作用域内计算模型预测值
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    
    # 计算损失相对于权重的梯度。输出gradients是一个列表，每个元素对应model.weights列表中的权重
    gradients = tape.gradient(average_loss, model.weights)
    
    # 利用梯度来更新权重
    update_weights(gradients, model.weights)

    return average_loss

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        # assign_sub相当于TensorFlow变量的-=
        w.assign_sub(g * learning_rate)

    return

'''
optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
'''