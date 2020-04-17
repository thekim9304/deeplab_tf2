from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, BatchNormalization, ReLU


class ResNet(Model):
    def __init__(self, n_name='50', out_stride=16, cycles=None):
        super(ResNet, self).__init__()

        if n_name is '18':
            cycles = [2, 2, 2, 2]
            bottle = False
        elif n_name is '34':
            cycles = [3, 4, 6, 3]
            bottle = False
        elif n_name is '50':
            cycles = [3, 4, 6, 3]
            bottle = True
        elif n_name is '101':
            cycles = [3, 4, 23, 3]
            bottle = True
        elif n_name is '152':
            cycles = [3, 8, 36, 3]
            bottle = True
        else:
            print('{} is unsuitable, ex){}, {}, {}, {}, {}'.format(n_name, '18', '34', '50', '101', '152'))
            raise NotImplemented

        if out_stride is 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif out_stride is 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
        else:
            print('{} is unsuitable, ex){}, {}'.format(out_stride, 16, 8))
            raise NotImplemented

        self.conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.mp = MaxPool2D((3, 3), 2, padding='same')

        self.block1 = _ResNetBlock(depth=64, cycle=cycles[0], bottle=bottle, stride=strides[0], rate=rates[0])
        self.block2 = _ResNetBlock(depth=128, cycle=cycles[1], bottle=bottle, stride=strides[1], rate=rates[1])
        self.block3 = _ResNetBlock(depth=256, cycle=cycles[2], bottle=bottle, stride=strides[2], rate=rates[2])
        self.block4 = _ResNetBlock(depth=512, cycle=cycles[3], bottle=bottle, stride=strides[3], rate=rates[3])

    def call(self, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.mp(x)
        x = self.block1(x)
        low_feature = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x, low_feature


class _ResNetBlock(Model):
    def __init__(self, depth, cycle, bottle, stride=1, rate=1):
        super(_ResNetBlock, self).__init__()

        if stride != 1 and rate < 2:
            f_stride = (2, 2)
        else:
            f_stride = (1, 1)

        if bottle:
            depths = [depth, depth, depth*4]
            kernels = [(1, 1), (3, 3), (1, 1)]
        else:
            depths = [depth, depth]
            kernels = [(3, 3), (3, 3)]

        self.layer_list = []
        for i in range(cycle):
            for j in range(len(depths)):
                # 블록의 첫 번째 이면
                if i == 0 and j == 0:
                    strides = f_stride
                    d_rate = (rate, rate)
                else:
                    if rate != 1:
                        strides = (stride, stride)
                        d_rate = (rate, rate)
                    else:
                        strides = (stride, stride)
                        d_rate = (rate, rate)
                self.layer_list.append(Conv2D(depths[j],
                                              kernel_size=kernels[j],
                                              strides=strides,
                                              padding='same',
                                              dilation_rate=d_rate))
                self.layer_list.append(BatchNormalization())
                self.layer_list.append(ReLU())

        self.dconv1 = Conv2D(depths[-1], (1, 1), f_stride, padding='same', activation='relu')
        self.add = Add()

    def call(self, x, name=None, *args):
        in_feature = x

        for layer in self.layer_list:
            x = layer(x)

        in_feature = self.dconv1(in_feature)
        x = self.add([in_feature, x])

        return x