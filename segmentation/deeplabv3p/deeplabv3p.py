from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, BatchNormalization, ReLU

from segmentation.deeplabv3p.decoder import Decoder
from segmentation.deeplabv3p.encoder import Encoder
from segmentation.deeplabv3p.backbone.resnet import ResNet


class Deeplabv3(Model):
    def __init__(self, class_num):
        super(Deeplabv3, self).__init__()

        self.backbone = ResNet()
        self.decoder = Decoder(class_num)
        self.encoder = Encoder()

    def call(self, x, *args):
        x, low_feature = self.backbone(x)
        en_feat = self.encoder(x)
        de_feat = self.decoder([low_feature, en_feat])

        return de_feat


def main():
    in_x = Input(shape=(128, 128, 3))

    model = Deeplabv3(10)

    print(model(in_x))


if __name__ == '__main__':
    main()
