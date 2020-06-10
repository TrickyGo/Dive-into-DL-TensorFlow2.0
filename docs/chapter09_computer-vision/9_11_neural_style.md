
# 9.11 样式迁移


```python
# Install TensorFlow
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import os
%matplotlib inline
```

如果你是一位摄影爱好者，也许接触过滤镜。它能改变照片的颜色样式，从而使风景照更加锐利或者令人像更加美白。但一个滤镜通常只能改变照片的某个方面。如果要照片达到理想中的样式，经常需要尝试大量不同的组合，其复杂程度不亚于模型调参。

在本节中，我们将介绍如何使用卷积神经网络自动将某图像中的样式应用在另一图像之上，即样式迁移（style transfer）[1]。这里我们需要两张输入图像，一张是内容图像，另一张是样式图像，我们将使用神经网络修改内容图像使其在样式上接近样式图像。图9.12中的内容图像为本书作者在西雅图郊区的雷尼尔山国家公园（Mount Rainier National Park）拍摄的风景照，而样式图像则是一幅主题为秋天橡树的油画。最终输出的合成图像在保留了内容图像中物体主体形状的情况下应用了样式图像的油画笔触，同时也让整体颜色更加鲜艳。

![alt text](https://zh.d2l.ai/_images/style-transfer.svg)

图 9.12 输入内容图像和样式图像，输出样式迁移后的合成图像

## 9.11.1. 方法

图9.13用一个例子来阐述基于卷积神经网络的样式迁移方法。首先，我们初始化合成图像，例如将其初始化成内容图像。该合成图像是样式迁移过程中唯一需要更新的变量，即样式迁移所需迭代的模型参数。然后，我们选择一个预训练的卷积神经网络来抽取图像的特征，其中的模型参数在训练中无须更新。深度卷积神经网络凭借多个层逐级抽取图像的特征。我们可以选择其中某些层的输出作为内容特征或样式特征。以图9.13为例，这里选取的预训练的神经网络含有3个卷积层，其中第二层输出图像的内容特征，而第一层和第三层的输出被作为图像的样式特征。接下来，我们通过正向传播（实线箭头方向）计算样式迁移的损失函数，并通过反向传播（虚线箭头方向）迭代模型参数，即不断更新合成图像。样式迁移常用的损失函数由3部分组成：内容损失（content loss）使合成图像与内容图像在内容特征上接近，样式损失（style loss）令合成图像与样式图像在样式特征上接近，而总变差损失（total variation loss）则有助于减少合成图像中的噪点。最后，当模型训练结束时，我们输出样式迁移的模型参数，即得到最终的合成图像。

![alt text](https://zh.d2l.ai/_images/neural-style.svg)

图 9.13 基于卷积神经网络的样式迁移。实线箭头和虚线箭头分别表示正向传播和反向传播

下面，我们通过实验来进一步了解样式迁移的技术细节。实验需要用到一些导入的包或模块。


```python
import time
import sys
sys.path.append("..")
```

## 9.11.2 读取内容图像和样式图像

首先，我们分别读取内容图像和样式图像。从打印出的图像坐标轴可以看出，它们的尺寸并不一样。


```python
content_img = tf.io.read_file('../../data/rainier.jpg')
content_img = tf.image.decode_jpeg(content_img)
plt.imshow(content_img)
```




    <matplotlib.image.AxesImage at 0x7f6bf001cba8>




![png](../img/chapter09/9.11/output_12_1.png)



```python
style_img = tf.io.read_file('../../data/autumn_oak.jpg')
style_img = tf.image.decode_jpeg(style_img)
plt.imshow(style_img)
```




    <matplotlib.image.AxesImage at 0x7f6b9536aa20>




![png](../img/chapter09/9.11/output_13_1.png)


## 9.11.3 预处理和后处理图像

下面定义图像的预处理函数和后处理函数。预处理函数preprocess对先对更改输入图像的尺寸，然后再将PIL图片转成卷积神经网络接受的输入格式，再在RGB三个通道分别做标准化，由于预训练模型是在均值为[0.485, 0.456, 0.406]标准差为[0.229, 0.224, 0.225]的图片数据上预训练的，所以我们要将图片标准化保持相同的均值和标准差。后处理函数postprocess则将输出图像中的像素值还原回标准化之前的值。由于图像每个像素的浮点数值在0到1之间，我们使用clamp函数对小于0和大于1的值分别取0和1。


```python
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img_tensor, image_shape):
    """
    image_shape = (h, w)
    """
    img = tf.image.resize(img_tensor, image_shape[0:2])
    img = tf.divide(img, 255.)
    img = tf.divide(tf.subtract(img, rgb_mean), rgb_std)
    img = tf.expand_dims(img, axis=0) # (batch_size, h, w, 3)
    return img

def postprocess(img_tensor):
    mean = -rgb_mean / rgb_std
    std= 1 / rgb_std
    img = tf.divide(tf.subtract(img_tensor, mean), std)
    return img
```

## 9.11.4 抽取特征

我们使用基于ImageNet数据集预训练的VGG-19模型来抽取图像特征 [1]。


```python
pretrained_net = keras.applications.vgg19.VGG19(weights="imagenet")
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5
    574717952/574710816 [==============================] - 7s 0us/step


为了抽取图像的内容特征和样式特征，我们可以选择VGG网络中某些层的输出。一般来说，越靠近输入层的输出越容易抽取图像的细节信息，反之则越容易抽取图像的全局信息。为了避免合成图像过多保留内容图像的细节，我们选择VGG较靠近输出的层，也称内容层，来输出图像的内容特征。我们还从VGG中选择不同层的输出来匹配局部和全局的样式，这些层也叫样式层。在5.7节（使用重复元素的网络（VGG））中我们曾介绍过，VGG网络使用了5个卷积块。实验中，我们选择第四卷积块的最后一个卷积层作为内容层，以及每个卷积块的第一个卷积层作为样式层。这些层的索引可以通过打印pretrained_net实例来获取


```python
need = ['name', 'filters', 'kernel_size', 'strides', 'pool_size', 'padding', 'batch_input_shape']
print("{}".format("\t"), end="")
for n in need:
    print(n.ljust(13,' '), end='\t')
print()
for i, layer in enumerate(pretrained_net.layers[:]):
    config = layer.get_config()
    params = [config.get(key) for key in need]
    print("({})".format(i), end="")
    for p in params:
        print(str(p).ljust(13,' '), end='\t')
    print()
```

    	name         	filters      	kernel_size  	strides      	pool_size    	padding      	batch_input_shape	
    (0)input_1      	None         	None         	None         	None         	None         	(None, 224, 224, 3)	
    (1)block1_conv1 	64           	(3, 3)       	(1, 1)       	None         	same         	None         	
    (2)block1_conv2 	64           	(3, 3)       	(1, 1)       	None         	same         	None         	
    (3)block1_pool  	None         	None         	(2, 2)       	(2, 2)       	valid        	None         	
    (4)block2_conv1 	128          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (5)block2_conv2 	128          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (6)block2_pool  	None         	None         	(2, 2)       	(2, 2)       	valid        	None         	
    (7)block3_conv1 	256          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (8)block3_conv2 	256          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (9)block3_conv3 	256          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (10)block3_conv4 	256          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (11)block3_pool  	None         	None         	(2, 2)       	(2, 2)       	valid        	None         	
    (12)block4_conv1 	512          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (13)block4_conv2 	512          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (14)block4_conv3 	512          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (15)block4_conv4 	512          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (16)block4_pool  	None         	None         	(2, 2)       	(2, 2)       	valid        	None         	
    (17)block5_conv1 	512          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (18)block5_conv2 	512          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (19)block5_conv3 	512          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (20)block5_conv4 	512          	(3, 3)       	(1, 1)       	None         	same         	None         	
    (21)block5_pool  	None         	None         	(2, 2)       	(2, 2)       	valid        	None         	
    (22)flatten      	None         	None         	None         	None         	None         	None         	
    (23)fc1          	None         	None         	None         	None         	None         	None         	
    (24)fc2          	None         	None         	None         	None         	None         	None         	
    (25)predictions  	None         	None         	None         	None         	None         	None         	



```python
# style_layers = [block1_conv1, block2_conv1, block3_conv1, 
#         block3_conv4, block4_conv1, block5_conv1]
# content_layers = [block3_conv1]
style_layers, content_layers = [1, 4, 7, 10, 12, 17], [7]
```

在抽取特征时，我们只需要用到VGG从输入层到最靠近输出层的内容层或样式层之间的所有层。下面构建一个新的网络net，它只保留需要用到的VGG的所有层。我们将使用net来抽取特征。


```python
net_list = []
for i in range(max(content_layers + style_layers) + 1):
    net_list.append(pretrained_net.layers[i])
net = keras.Sequential()
for layer in net_list:
        net.add(layer)
```


```python
net.layers[1].name
```




    'block1_conv2'



给定输入X，如果简单调用前向计算net(X)，只能获得最后一层的输出。由于我们还需要中间层的输出，因此这里我们逐层计算，并保留内容层和样式层的输出。


```python
def extract_feature(x, content_layers, style_layers):
    assert len(x.shape) == 4

    contents = []
    styles = []
    for i, layer in enumerate(net.layers):
        x = layer(x)
        # 因为input层没有算（我也不知道为啥没算
        if i+1 in style_layers:
            styles.append(x)
        if i+1 in content_layers:
            contents.append(x)
    return contents, styles
```

下面定义两个函数，其中get_contents函数对内容图像抽取内容特征，而get_styles函数则对样式图像抽取样式特征。因为在训练时无须改变预训练的VGG的模型参数，所以我们可以在训练开始之前就提取出内容图像的内容特征，以及样式图像的样式特征。由于合成图像是样式迁移所需迭代的模型参数，我们只能在训练过程中通过调用extract_features函数来抽取合成图像的内容特征和样式特征。


```python
def get_contents(image_shape):
    # 统一尺寸
    content_x = preprocess(content_img, image_shape)
    # 获得内容特征
    contents_y, _ = extract_feature(content_x, content_layers, style_layers)
    return content_x, contents_y

def get_styles(image_shape):
    # 统一尺寸
    style_x = preprocess(style_img, image_shape)
    # 获得样式特征
    _, styles_y = extract_feature(style_x, content_layers, style_layers)
    return style_x, styles_y
```

## 9.11.5 定义损失函数

下面我们来描述样式迁移的损失函数。它由内容损失、样式损失和总变差损失3部分组成。

### 9.11.5.1 内容损失

与线性回归中的损失函数类似，内容损失通过平方误差函数衡量合成图像与内容图像在内容特征上的差异。平方误差函数的两个输入均为extract_features函数计算所得到的内容层的输出。


```python
def content_loss(y_hat, y):
    c_l = keras.losses.mean_squared_error(y, y_hat)
    return tf.reduce_mean(c_l)
```

### 9.11.5.2 样式损失

样式损失也一样通过平方误差函数衡量合成图像与样式图像在样式上的差异。为了表达样式层输出的样式，我们先通过extract_features函数计算样式层的输出。假设该输出的样本数为1，通道数为 c ，高和宽分别为 h 和 w ，我们可以把输出变换成 c 行 hw 列的矩阵 X 。矩阵 X 可以看作由 c 个长度为 hw 的向量 x1,…,xc 组成的。其中向量 xi 代表了通道 i 上的样式特征。这些向量的格拉姆矩阵（Gram matrix） XX⊤∈Rc×c 中 i 行 j 列的元素 xij 即向量 xi 与 xj 的内积，它表达了通道 i 和通道 j 上样式特征的相关性。我们用这样的格拉姆矩阵表达样式层输出的样式。需要注意的是，当 hw 的值较大时，格拉姆矩阵中的元素容易出现较大的值。此外，格拉姆矩阵的高和宽皆为通道数 c 。为了让样式损失不受这些值的大小影响，下面定义的gram函数将格拉姆矩阵除以了矩阵中元素的个数，即 chw 。


```python
def gram(x):
    # x.shape=(batch_size, h, w, c)
    num_channels, n = x.shape[3], x.shape[1] * x.shape[2]
    x = tf.reshape(x, shape=(num_channels, n))
    x_big = tf.matmul(x, tf.transpose(x))
    x_big = tf.divide(x_big, num_channels * n)
    return x_big
```

自然地，样式损失的平方误差函数的两个格拉姆矩阵输入分别基于合成图像与样式图像的样式层输出。这里假设基于样式图像的格拉姆矩阵gram_Y已经预先计算好了。


```python
def style_loss(y_hat, gram_y):
    s_l = keras.losses.mean_squared_error(gram_y, gram(y_hat))
    return tf.reduce_mean(s_l)
```


```python
x = np.array([[[[1],[2],[3]],[[2],[3],[3]]]])
y = tf.ones((1,1))
style_loss(x, y)
# x.shape
```




    <tf.Tensor: shape=(), dtype=float64, numpy=25.0>



### 9.11.5.3 总变差损失

我们学到的合成图像里面有大量高频噪点，即有特别亮或者特别暗的颗粒像素。一种常用的降噪方法是总变差降噪（total variation denoising）。假设 xi,j 表示坐标为 (i,j) 的像素值，降低总变差损失

$\sum_{i,j} \left|x_{i,j} - x_{i+1,j}\right| + \left|x_{i,j} - x_{i,j+1}\right|$

能够尽可能使邻近的像素值相似。


```python
def tv_loss(y_hat):
    t1 = keras.losses.mean_absolute_error(y_hat[:, 1:, :, :], 
                        y_hat[:, :-1, :, :])
    t1 = tf.reduce_mean(t1)
    t2 = keras.losses.mean_absolute_error(y_hat[:, :, 1:, :], 
                        y_hat[:, :, :-1, :])
    t2 = tf.reduce_mean(t2)
    
    tv = tf.add(t1, t2)
    return tf.multiply(0.5, tv)
```

## 9.11.5.4 损失函数

样式迁移的损失函数即内容损失、样式损失和总变差损失的加权和。通过调节这些权值超参数，我们可以权衡合成图像在保留内容、迁移样式以及降噪三方面的相对重要性。


```python
# 超参数
content_weight, style_weight, tv_weight = 1, 1e3, 100

def compute_loss(x, contents_y_hat, styles_y_hat, contents_y, styles_y_gram):
    # 分别计算内容损失、样式损失和总变差损失
    contents_l = [content_loss(y_hat, y) * content_weight for y_hat, y in zip(
        contents_y_hat, contents_y)]
    styles_l = [style_loss(y_hat, y) * style_weight for y_hat, y in zip(
        styles_y_hat, styles_y_gram)]
    tv_l = tv_loss(x) * tv_weight
    # 对所有损失求和
    l = tf.reduce_sum(tf.reshape(contents_l,(-1,))) + tf.reduce_sum(tf.reshape(styles_l,(-1,))) + tv_l
    return contents_l, styles_l, tv_l, l
```

### 9.11.6 创建和初始化合成图像

在样式迁移中，合成图像是唯一需要更新的变量。因此，我们可以定义一个简单的模型GeneratedImage，并将合成图像视为模型参数。模型的前向计算只需返回模型参数即可。


```python
class GeneratedImage(keras.layers.Layer):
    def __init__(self, img):
        super().__init__()
        self.weight = tf.Variable(img, dtype=tf.float32)
    
    def __call__(self):
        return self.weight
```

下面，我们定义get_inits函数。该函数创建了合成图像的模型实例，并将其初始化为图像X。样式图像在各个样式层的格拉姆矩阵styles_Y_gram将在训练前预先计算好。


```python
def get_inits(x, lr, styles_y):
    gen_img = GeneratedImage(x)
    # 这里如果像pytorch那样直接赋值会报错，所以我在上面那个代码赋值的
    # gen_img.weight = x
    optimizer = keras.optimizers.Adam(lr)
    styles_y_gram = [gram(y) for y in styles_y]
    return gen_img(), styles_y_gram, optimizer
```

## 9.11.7 训练

在训练模型时，我们不断抽取合成图像的内容特征和样式特征，并计算损失函数。


```python

def train(x, contents_y, styles_y, lr, max_epochs, lr_decay_epoch):
    x, styles_y_gram, optimizer = get_inits(x, lr, styles_y)

    for i in range(max_epochs):
        start = time.time()
        with tf.GradientTape() as t:
            t.watch(x)
            contents_y_hat, styles_y_hat = extract_feature(x, content_layers,
                                style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(x,
                                contents_y_hat,
                                styles_y_hat,
                                contents_y,
                                styles_y_gram)
            
        grads = t.gradient(l, x)
        optimizer.apply_gradients(zip([grads], [x]))


        if i % 100 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, tf.reduce_sum(contents_l), 
                     tf.reduce_sum(styles_l), 
                     tf.reduce_sum(tv_l),
                     time.time() - start))
    return x
```

下面我们开始训练模型。首先将内容图像和样式图像的高和宽分别调整为150和225像素。合成图像将由内容图像来初始化。


```python
import time
image_shape =  (150, 225, 3)
content_x, contents_y = get_contents(image_shape)
style_x, styles_y = get_styles(image_shape)
output = train(content_x, contents_y, styles_y, 0.1, 501, 100)
```

    epoch 100, content loss 34.57, style loss 37.54, TV loss 32.21, 0.03 sec
    epoch 200, content loss 29.26, style loss 33.96, TV loss 22.23, 0.03 sec
    epoch 300, content loss 28.32, style loss 32.54, TV loss 19.48, 0.03 sec
    epoch 400, content loss 27.88, style loss 32.21, TV loss 18.42, 0.03 sec
    epoch 500, content loss 28.00, style loss 29.96, TV loss 17.93, 0.03 sec



```python
img = postprocess(output)[0]
img = tf.clip_by_value(img, 0, 1)
plt.imsave('../../data/1.jpg', img.numpy())
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x7f6b9f739978>




![png](../img/chapter09/9.11/output_57_1.png)


为了得到更加清晰的合成图像，下面我们在更大的[Math Processing Error]300×450尺寸上训练。我们将图9.14的高和宽放大2倍，以初始化更大尺寸的合成图像。


```python
image_shape =  (300, 450, 3)
content_x, contents_y = get_contents(image_shape)
style_x, styles_y = get_styles(image_shape)
big_output = train(content_x, contents_y, styles_y, 0.01, 1001, 100)
```

    epoch 100, content loss 17.56, style loss 43.69, TV loss 24.68, 0.09 sec
    epoch 200, content loss 17.06, style loss 35.46, TV loss 23.24, 0.09 sec
    epoch 300, content loss 16.64, style loss 32.35, TV loss 21.74, 0.09 sec
    epoch 400, content loss 16.37, style loss 30.59, TV loss 20.40, 0.09 sec
    epoch 500, content loss 16.21, style loss 29.39, TV loss 19.29, 0.09 sec
    epoch 600, content loss 16.10, style loss 28.52, TV loss 18.37, 0.09 sec
    epoch 700, content loss 16.02, style loss 27.85, TV loss 17.61, 0.09 sec
    epoch 800, content loss 15.95, style loss 27.32, TV loss 16.99, 0.09 sec
    epoch 900, content loss 15.90, style loss 26.88, TV loss 16.47, 0.09 sec
    epoch 1000, content loss 15.84, style loss 26.51, TV loss 16.04, 0.09 sec



```python
img=postprocess(big_output)[0]
img = tf.clip_by_value(img, 0, 1)
plt.imsave('../../data/2.jpg', img.numpy())
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x7f6b9f4ea0b8>




![png](../img/chapter09/9.11/output_60_1.png)



## 9.11.8 小结

* 样式迁移常用的损失函数由3部分组成：内容损失使合成图像与内容图像在内容特征上接近，样式损失令合成图像与样式图像在样式特征上接近，而总变差损失则有助于减少合成图像中的噪点。
* 可以通过预训练的卷积神经网络来抽取图像的特征，并通过最小化损失函数来不断更新合成图像。
* 用格拉姆矩阵表达样式层输出的样式。


## 9.11.9 练习

* 选择不同的内容和样式层，输出有什么变化？
* 调整损失函数中的权值超参数，输出是否保留更多内容或减少更多噪点？
* 替换实验中的内容图像和样式图像，你能创作出更有趣的合成图像吗？


