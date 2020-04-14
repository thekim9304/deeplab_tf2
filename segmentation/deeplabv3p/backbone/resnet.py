from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Add, GlobalAveragePooling2D


class ResNet(Model):
    def __init__(self, layer_num=2, cycles=None):
        super(ResNet, self).__init__()

        if cycles is None:
            cycles = [2, 2, 2, 2]

        self.conv = Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        self.mp = MaxPool2D((3, 3), 2)
        self.block1 = _ResNetBlock(n_layer=layer_num, cycle=cycles[0], depth=64, d_sampling=False)
        self.block2 = _ResNetBlock(n_layer=layer_num, cycle=cycles[1], depth=128)
        self.block3 = _ResNetBlock(n_layer=layer_num, cycle=cycles[2], depth=256)
        self.block4 = _ResNetBlock(n_layer=layer_num, cycle=cycles[3], depth=512)

    def call(self, x, *args, **kwargs):
        x = self.conv(x)
        x = self.mp(x)
        x = self.block1(x)
        x = self.block2(x)
        low_feature = x
        x = self.block3(x)
        x = self.block4(x)

        return x, low_feature


class _ResNetBlock(Model):
    def __init__(self, n_layer, cycle, depth, d_sampling=True):
        super(_ResNetBlock, self).__init__()

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


