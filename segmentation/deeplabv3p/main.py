from segmentation.deeplabv3p.model.deeplabv3p import Deeplabv3
from utils.load_data import img_size_check
import tensorflow as tf

import cv2
import numpy as np


def main():
    img_path = './data/image/000000_10.png'
    ins_path = './data/instance/000000_10.png'

    img = cv2.imread(img_path)
    img = img_size_check(img)

    img = img[tf.newaxis, ...]
    img = img.astype(np.float32)

    y = cv2.imread(ins_path, 0)
    y = img_size_check(y)
    y = y[tf.newaxis, ...]
    y = y.astype(np.float32)

    model = Deeplabv3(class_num=2)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])


    # model.build(input_shape=img.shape)
    #
    # model.fit(img, y, epochs=100)

    # predict = model(img)
    #
    # predict = tf.squeeze(predict, 0)


if __name__ == '__main__':
    main()