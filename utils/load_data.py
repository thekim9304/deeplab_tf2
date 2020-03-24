import tensorflow as tf


def load_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 채널 차원을 추가합니다.
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # print('x_train.shape : {}, x_test.shape : {}'.format(x_train.shape, x_test.shape))

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # return train_ds, test_ds
    return x_train, y_train
