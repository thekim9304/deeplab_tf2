from tensorflow.keras.preprocessing.image import ImageDataGenerator

# segmentation 모델에 입력될 이미지 사이즈는 32로 나누어 떨어져야함 (32*n, 32*n)


def load_data_to_segmentation(root_path, data_gen_args, img_size=(256, 256), batch_size=32, seed=1):
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow_from_directory(root_path + '/images',
                                                        target_size=img_size,
                                                        color_mode='rgb',
                                                        class_mode=None,
                                                        batch_size=batch_size,
                                                        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(root_path + '/masks',
                                                      target_size=img_size,
                                                      color_mode='grayscale',
                                                      class_mode=None,
                                                      batch_size=batch_size,
                                                      seed=seed)

    return image_generator, mask_generator


# example
#     ㄴ segmentation
#         ㄴ images
#             ㄴ img
#                 ㄴ image1.png
#                 ㄴ image2.png
#         ㄴ masks
#             ㄴ img
#                 ㄴ mask1.png
#                 ㄴ mask2.png
def main():
    root_path = '../deeplabv3p/data'
    data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.3,
                         horizontal_flip=True)
    img_size = (224, 224)
    batch_size = 32
    epochs = 5

    # Case 1  : use_generator=True
    # image_generator, mask_generator = load_data_to_segmentation(root_path=root_path,
    #                                                      data_gen_args=data_gen_args,
    #                                                      img_size=img_size,
    #                                                      batch_size=batch_size,
    #                                                      seed=50)
    # model.fit_generator(
    #     zip(image_generator, mask_generator),
    #     steps_per_epoch=len(image_generator),
    #     epochs=50)

    # Case 2  : use_generator=True
    image_generator, mask_generator = load_data_to_segmentation(root_path=root_path,
                                                                data_gen_args=data_gen_args,
                                                                img_size=img_size,
                                                                batch_size=batch_size,
                                                                seed=50)
    for epoch in range(epochs):
        print('Epoch : ', epoch)

        for batch in range(len(image_generator)):
            images, labels = next(image_generator), next(mask_generator)
            # model.fit(images, labels)
            print(images.shape, labels.shape)


if __name__ == '__main__':
    main()