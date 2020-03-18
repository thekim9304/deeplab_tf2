import tensorflow as tf
import ipykernel
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Add, GlobalAveragePooling2D

import utils.load_data as ld

import cv2
import numpy as np


class ResNetBlock(Model):
    def __init__(self, n_layer, cycle, depth, d_sampling=True):
        super(ResNetBlock, self).__init__()

        if d_sampling:
            f_stride = (2, 2)
        else:
            f_stride = (1, 1)

        if n_layer == 3:
            depths = [depth, depth, depth*4]
            kernels = [(1, 1), (3, 3), (1, 1)]
        else:
            depths = [depth, depth]
            kernels = [(3, 3), (3, 3)]

        self.layer_list = []
        for i in range(cycle):
            for j in range(n_layer):
                # 블록의 첫 번째 이면
                if i == 0 and j == 0:
                    stride = f_stride
                else:
                    stride = (1, 1)

                self.layer_list.append(Conv2D(depths[j], kernel_size=kernels[j], strides=stride, padding='same', activation='relu'))

        self.dconv1 = Conv2D(depths[-1], (1, 1), f_stride, padding='same', activation='relu')
        self.add = Add()

    def call(self, x, name=None, *args):
        in_feature = x

        for layer in self.layer_list:
            x = layer(x)

        in_feature = self.dconv1(in_feature)
        x = self.add([in_feature, x])

        return x


class ResNet(Model):
    def __init__(self, class_num=1000, layer_num=2, cycles=None):
        super(ResNet, self).__init__()

        if cycles is None:
            cycles = [2, 2, 2, 2]

        self.conv = Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        self.mp = MaxPool2D((3, 3), 2)
        self.block1 = ResNetBlock(n_layer=layer_num, cycle=cycles[0], depth=64, d_sampling=False)
        self.block2 = ResNetBlock(n_layer=layer_num, cycle=cycles[1], depth=128)
        self.block3 = ResNetBlock(n_layer=layer_num, cycle=cycles[2], depth=256)
        self.block4 = ResNetBlock(n_layer=layer_num, cycle=cycles[3], depth=512)
        self.gap = GlobalAveragePooling2D()
        self.predict = Dense(class_num, activation='softmax')

    def call(self, x, *args, **kwargs):
        x = self.conv(x)
        x = self.mp(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.predict(x)

        return x


def main():
    cycles = [3, 4, 6, 3]
    model = ResNet(10, 3, cycles)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    model.build(input_shape=(None, 24, 24, 1))

    train_ds, test_ds = ld.load_mnist()

    model.fit(train_ds, epochs=5)


if __name__ == '__main__':
    main()
