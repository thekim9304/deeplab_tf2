"""
- ResNet은 feature들이 layer로 전달되기 전에 summation(Add)을 통해 결합되지만,
    DenseNet은 feature들을 concatenation한다.
    : lth layer는 모든 선행 conv block의 feature-map들로 구성된 l개의 input을 가진다

- BN -> ReLu -> weight(conv) 순서

- Transition layer : 인접한 두 block 사이의 layer, convolution과 pooling을 통해 feature map의 크기를 변경
    + 모델을 보다 소형으로 만들기 위해, transition layer에서 feature-map의 개수를 줄인다.
        : Dense block이 m개의 feature-map을 포함하는 경우, 뒤따르는 transition layer에서 출력 feature-map을 [θ
m]개 생성한다.
        : 여기서 0 < θ < 1은 'compression factor'라고 한다.

- Growth rate : 각 함수 Hl()은 k개의 feature map을 생성한다.
    : 여기서 hyperparameter k를 네트워크의 'growth rate'라고 한다.
"""

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Activation,\
    Flatten, Dense, Input, Concatenate, GlobalAveragePooling2D

import ipykernel

import utils.load_data as ld


class _ConvBlock(Model):
    def __init__(self, growth_rate, activation):
        super(_ConvBlock, self).__init__()

        self.bn1 = BatchNormalization()
        self.acti1 = Activation(activation=activation)
        self.conv1 = Conv2D(4 * growth_rate, 1, padding='same')
        self.bn2 = BatchNormalization()
        self.acti2 = Activation(activation=activation)
        self.conv2 = Conv2D(growth_rate, 3, padding='same')
        self.concat = Concatenate()

    def call(self, x, *args):
        x_t = self.bn1(x)
        x_t = self.acti1(x_t)
        x_t = self.conv1(x_t)
        x_t = self.bn2(x_t)
        x_t = self.acti2(x_t)
        x_t = self.conv2(x_t)

        return self.concat([x, x_t])


class _DenseBlock(Model):
    def __init__(self, blocks, growth_rate, activation='relu'):
        super(_DenseBlock, self).__init__()

        self.layer_list = []
        for _ in range(blocks):
            self.layer_list.append(_ConvBlock(growth_rate, activation))

    def call(self, x, *args):
        for layer in self.layer_list:
            x = layer(x)

        return x


class _TransitionLayer(Model):
    def __init__(self, depth, compression_factor=0.5, activation='relu'):
        super(_TransitionLayer, self).__init__()

        self.bn = BatchNormalization()
        self.activation = Activation(activation)
        self.conv = Conv2D(int(depth * compression_factor), (1, 1), padding='same')
        self.avgpool = AvgPool2D((2, 2), padding='same', strides=2)

    def call(self, x, *args):
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.avgpool(x)

        return x


class DenseNet(Model):
    def __init__(self, class_num=1000, num_in_block=None, growth_rate=32, compression_factor=0.5, activation='relu'):
        super(DenseNet, self).__init__()

        if num_in_block is None:
            num_in_block = [6, 12, 24, 16]

        self.conv1 = Conv2D(growth_rate * 2, (7, 7), padding='same', strides=2, activation=activation)
        self.maxpool = MaxPool2D((3, 3), strides=2, padding='same')

        self.blocks = Sequential()
        for i in range(4):
            self.blocks.add(_DenseBlock(num_in_block[i], growth_rate, activation))
            if i != 3:
                self.blocks.add(_TransitionLayer(growth_rate, compression_factor, activation))

        self.gap = GlobalAveragePooling2D()
        self.predict = Dense(class_num, activation='softmax')

    def call(self, x, *args):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = self.predict(x)

        return x


def main():
    model = DenseNet(10)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['sparse_categorical_accuracy'])

    x = Input((224, 224, 3))
    x = model(x)

    print(x.shape)


if __name__ == '__main__':
    main()