#user：lixiangyang
import tensorflow as tf
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
#我们生成与上一节中相同的数据集。其中features是训练数据特征，labels是标签
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal([num_examples, num_inputs],mean=0,stddev=1)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += tf.random.normal(labels.shape,0,1)
print(features[0], labels[0])


# 读取数据
dataset=tf.data.Dataset.from_tensor_slices((features,labels))
train_db = tf.data.Dataset.from_tensor_slices((features, labels)).batch(10)
#train_db = tf.data.Dataset.from_tensor_slices((features, labels)).batch(32)

for(x,y) in train_db:
    print(x,y)


#定义模型,tensorflow 2.0推荐使用keras定义网络，故使用keras定义网络
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Dense(1))

#定义损失函数和优化器：损失函数为mse，优化器选择sgd随机梯度下降def loss_fn(model, x, y):
model.compile(optimizer=tf.keras.optimizers.SGD(0.03),
              loss='mse')


#训练模型
model.fit(features,labels,epochs=3, batch_size=10)

#获取w和b的参数
true_w, model.get_weights()[0]
true_b, model.get_weights()[1]
