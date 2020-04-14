from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, BatchNormalization, ReLU


class Decoder(Model):
    def __init__(self, class_num):
        super(Decoder, self).__init__()
        self._1conv = Sequential([Conv2D(256, 1, padding='same'),
                                  BatchNormalization(),
                                  ReLU()])
        self.upsampling1 = UpSampling2D((4, 4))
        self.concat = Concatenate()
        self._3conv1 = Sequential([Conv2D(256, 3, padding='same'),
                                   BatchNormalization(),
                                   ReLU()])
        self._3conv2 = Sequential([Conv2D(256, 3, padding='same'),
                                   BatchNormalization(),
                                   ReLU()])
        self._3conv3 = Sequential([Conv2D(class_num, 3, padding='same'),
                                   BatchNormalization(),
                                   ReLU()])
        self.upsampling2 = UpSampling2D((4, 4))

    def call(self, features, *args):
        x = self._1conv(features[0])
        x = self.concat([x, self.upsampling1(features[1])])
        x = self._3conv1(x)
        x = self._3conv2(x)
        x = self._3conv3(x)

        return self.upsampling2(x)


def main():
    in_x = Input(shape=(128, 128, 3))
    in_x2 = Input(shape=(32, 32, 3))

    decoder = Decoder(10)

    print(decoder([in_x, in_x2]))


if __name__ == '__main__':
    main()
