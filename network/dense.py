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


class ConvBlock(Model):
    def __init__(self):
        super(ConvBlock, self).__init__()



class DenseBlock(Model):
    def __init__(self, cycle, growth_rate, activation='relu'):
        super(DenseBlock, self).__init__()

        self.layer_list = []
        for _ in range(cycle):
            for i in range(2):
                if i == 0:
                    g_rate = growth_rate * 4
                    filters = (1, 1)
                else:
                    g_rate = growth_rate
                    filters = (3, 3)

                self.layer_list.append(BatchNormalization())
                self.layer_list.append(Activation(activation))
                self.layer_list.append(Conv2D(g_rate, filters, padding='same'))

        self.concat = Concatenate()

    def call(self, x, *args):
        c_x = x
        for i, layer in enumerate(self.layer_list):
            if i % 2 is 0:
                c_x = x

            x = layer(x)

            if i % 2 is not 0:
                x = self.concat([c_x, x])

        return x



class TransitionLayer(Model):
    def __init__(self, depth, compression_factor=1, activation='relu'):
        super(TransitionLayer, self).__init__()

        self.bn = BatchNormalization()
        self.activation = Activation(activation)
        self.conv = Conv2D(depth * compression_factor, (1, 1), padding='same')
        self.avgpool = AvgPool2D((2, 2), padding='same', strides=2)

    def call(self, x, *args):
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.avgpool(x)
        print(x.shape)

        return x

class DenseNet(Model):
    def __init__(self, class_num=1000, num_in_block=None, growth_rate=32, compression_factor=1, activation='relu'):
        super(DenseNet, self).__init__()

        if num_in_block is None:
            num_in_block = [6, 12, 24, 16]

        self.conv1 = Conv2D(growth_rate * 2, (7, 7), padding='same', strides=2, activation=activation)
        self.maxpool = MaxPool2D((3, 3), strides=2, padding='same')

        self.blocks = Sequential()
        for i in range(4):
            self.blocks.add(DenseBlock(num_in_block[i] , growth_rate, activation))
            if i != 3:
                self.blocks.add(TransitionLayer(growth_rate, compression_factor, activation))

        self.gap = GlobalAveragePooling2D()
        self.predict = Dense(class_num, activation='softmax')

    def call(self, x, *args):
        print('in', x)
        x = self.conv1(x)
        print('conv1', x.shape)
        x = self.maxpool(x)
        print('mp1', x.shape)
        x = self.blocks(x)
        print('blocks', x.shape)
        x = self.gap(x)
        print('gap', x)
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