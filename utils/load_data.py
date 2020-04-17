import tensorflow as tf
import cv2

def img_size_check(img):
    h, w = img.shape[:2]
    modi_h = h + (4 - (h % 4))
    modi_w = w + (4 - (w % 4))

    return cv2.resize(img, (modi_w, modi_h))

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
