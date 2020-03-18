import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

import utils.load_data as ld


class ResNetBlock(Model):
    def __init__(self, layer_num, depth):
        super(ResNetBlock, self).__init__()

        # self.layer_list = []
        # for _ in range(layer_num):
        #     self.layer_list.append(Conv2D(depth, (3, 3), activation='relu'))
        # self.layer_list.append(MaxPool2D())

    def call(self, x, *args):
        # for layer in self.layer_list:
        #     x = layer(x)

        return x

class ResNet(Model):
    def __init__(self, class_num, block_list, depth_list):
        super(ResNet, self).__init__()

        # self.num_class = class_num
        # self.block_list = block_list
        # self.depth_list = depth_list
        #
        # self.blocks = Sequential()
        #
        # for layer_num, depth in zip(self.block_list, self.depth_list):
        #     self.blocks.add(ResNetBlock(layer_num, depth))
        #
        # self.flatten = Flatten()
        # self.d1 = Dense(4096, activation='relu')
        # self.d2 = Dense(4096, activation='relu')
        # self.predict = Dense(class_num, activation='softmax')

    def call(self, x, *args, **kwargs):
        # x = self.blocks(x)
        # x = self.flatten(x)
        # x = self.d1(x)
        # x = self.d2(x)
        # x = self.predict(x)

        return x



def main():
    model = ResNet(10, [2, 3], [64, 64])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    model.build(input_shape=(None, 28, 28, 1))

    x_train, y_train = ld.load_mnist()
    print(x_train.shape, y_train.shape)

    model.fit(x_train, y_train,
              batch_size=64,
              epochs=10)


if __name__ == '__main__':
    main()
