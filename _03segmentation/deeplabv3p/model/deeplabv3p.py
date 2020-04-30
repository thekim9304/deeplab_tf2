from tensorflow.keras import Model, Input

from _03segmentation.deeplabv3p.model.decoder import Decoder
from _03segmentation.deeplabv3p.model.encoder import Encoder
from _03segmentation.deeplabv3p.backbone.resnet import ResNet


class Deeplabv3(Model):
    def __init__(self, class_num, out_stride=16):
        super(Deeplabv3, self).__init__()

        self.backbone = ResNet(n_name='101', out_stride=out_stride)
        self.decoder = Decoder(class_num)
        self.encoder = Encoder()

    def call(self, x, *args):
        x, low_feature = self.backbone(x)
        en_feat = self.encoder(x)
        de_feat = self.decoder([low_feature, en_feat])

        return de_feat


def main():
    in_x = Input(shape=(224, 224, 3))

    model = Deeplabv3(2)

    out_feature = model(in_x)

    print(out_feature.shape)

if __name__ == '__main__':
    main()
