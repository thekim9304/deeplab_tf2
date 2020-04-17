from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pathlib
import numpy as np
import matplotlib.pyplot as plt


def load_data_to_classification(root_path, data_gen_args, img_size=(256, 256), batch_size=32, class_mode='categorical', seed=1):
    data_dir = pathlib.Path(root_path)
    image_count = len(list(data_dir.glob('*/*')))
    print('img_cnt : ', image_count)
    class_names = np.array([item.name for item in data_dir.glob('*')])
    print('class_names : ', class_names)

    image_datagen = ImageDataGenerator(**data_gen_args)

    return image_datagen.flow_from_directory(str(data_dir),
                                             target_size=img_size,
                                             class_mode=class_mode,
                                             batch_size=batch_size,
                                             seed=seed)

# return image_generator.flow_from_directory(directory=str(data_dir),
#                                            batch_size=batch_size,
#                                            shuffle=True,
#                                            target_size=size,
#                                            classes=list(class_names))

# cat&dog
#     ㄴ class1
#         ㄴ class1_img1.jpg
#         ㄴ class1_img2.jpg
#         ㄴ ...
#     ㄴ class2
#         ㄴ class2_img1.jpg
#         ㄴ class2_img2.jpg
#     ㄴ class3
#         ㄴ class3_img1.jpg
#         ㄴ class3_img2.jpg
#     ㄴ ...
def main():
    root_path = '../../data/cat&dog'
    data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.3,
                     horizontal_flip=True)
    img_size = (224, 224)
    batch_size = 3
    epochs = 5

    train_generator = load_data_to_classification(root_path=root_path,
                                                  data_gen_args=data_gen_args,
                                                  img_size=img_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  seed=1)

    # Case 1
    # model.fit_generator(train_generator,
    #                     steps_per_epoch=len(train_generator),
    #                     epochs=epochs)

    # Case 2
    for epoch in range(epochs):
        print('Epoch : ', epoch)

        for batch in range(len(train_generator)):
            images, labels = next(train_generator)
            # model.fit(images, labels)
            print(images.shape, labels.shape)


if __name__=='__main__':
    main()
