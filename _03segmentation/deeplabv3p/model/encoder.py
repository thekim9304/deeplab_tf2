from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Conv2D, Concatenate, BatchNormalization, AvgPool2D, ReLU

import tensorflow as tf


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self._1conv1 = Conv2D(256, 1, 1, padding='same', activation='relu')
        self._3conv6 = Conv2D(256, 3, padding='same', dilation_rate=(6, 6), activation='relu')
        self._3conv12 = Conv2D(256, 3, padding='same', dilation_rate=(12, 12), activation='relu')
        self._3conv18 = Conv2D(256, 3, padding='same', dilation_rate=(18, 18), activation='relu')
        self.img_pool = Sequential([AvgPool2D((1, 1)),
                                   Conv2D(256, 1),
                                   BatchNormalization(),
                                   tf.keras.layers.ReLU()])
        self.concat = Concatenate()
        self.last_feature = Sequential([Conv2D(256, 1, padding='same'),
                                        BatchNormalization(),
                                        ReLU()])

    def call(self, x, *args):
        x1 = self._1conv1(x)
        x2 = self._3conv6(x)
        x3 = self._3conv12(x)
        x4 = self._3conv18(x)
        x5 = self.img_pool(x)
        x = self.concat([x1, x2, x3, x4, x5])
        x = self.last_feature(x)

        return x


def main():
    in_x = Input(shape=(128, 128, 3))

    encoder = Encoder()

    print(encoder(in_x))


if __name__ == '__main__':
    main()
