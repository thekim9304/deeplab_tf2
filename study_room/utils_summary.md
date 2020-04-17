##pathlib
---
- pathlib 모듈의 기본 아이디어는 파일시스템 경로를 단순한 문자열이 아니라 객체로 다루자는 것이다.


##ImageDataGenerator
---
- [Data Augmentation Example URL](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)
- [Keras document](https://keras.io/preprocessing/image/)
- 학습 도중에 이미지에 임의 변형 및 정규화 적용
- 변형된 이미지를 배치 단위로 불러올 수 있는 generator 생성
  - generator를 생성할 때 flow(data, labels), flow_from_directory(directory) 두 가지 함수를 사용한다.
  - fit_generator, evaluate_generator 함수를 이용해 generator로 이미지를 불러와서 모델을 학습시킬 수 있다.

- **parameter**
  - rescale = 1./255
    - 이미지의 픽셀 값을 0~1로 정규화

  - rotation_range
    - 지정된 각도 범위내에서 임의로 원본이미지를 회전시킨다.
    - 예를 들어 90이면 0도에서 90도 사이에 임의의 각도로 회전시킨다.
    ![2017-3-8-CNN_Data_Augmentation_5_rotate](/assets/2017-3-8-CNN_Data_Augmentation_5_rotate.png)

  - width_shift_range
    - 지정된 수평방향 이동 범위내에서 임의로 원본이미지를 이동시킨다.
    - 수치는 전체 넓이의 비율(실수)로 나타낸다.
    - 예를 들어 0.1이고 전체 넓이가 100이면, 10픽셀 내외로 좌우 이동시킨다.
    ![2017-3-8-CNN_Data_Augmentation_5_width_shift](/assets/2017-3-8-CNN_Data_Augmentation_5_width_shift.png)

  - height_shift_range
    - 지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동시킨다.
    - 수치는 전체 높이의 비율(실수)로 나타낸다.
    - 예를 들어, 0.1이고 전체 높이가 100이면, 10픽셀 내외로 상하 이동시킨다.
    ![2017-3-8-CNN_Data_Augmentation_5_height_shift](/assets/2017-3-8-CNN_Data_Augmentation_5_height_shift_6pzsnspgx.png)

  - shear_range
    - 밀림 강도 범위내에서 임의로 원본이미지를 변형시킨다.
    - 수치는 시계 반대방향으로 밀림 강도를 라디안으로 나타낸다.
    - 예를 들어, 0.5이라면 0.5 라디안 내외로 시계 반대방향으로 변형시킨다.
    ![2017-3-8-CNN_Data_Augmentation_5_shear](/assets/2017-3-8-CNN_Data_Augmentation_5_shear.png)

  - zoom_range
    - 지정된 확대/축소 범위내에서 임의로 원본이미지를 확대/축소한다.
    - '1-수치'부터 '1+수치'사이 범위로 확대/축소를 한다.
    - 예를 들어, 0.3이라면 0.7배에서 1.3배 크기 변화를 시킨다.
    ![2017-3-8-CNN_Data_Augmentation_5_zoom](/assets/2017-3-8-CNN_Data_Augmentation_5_zoom.png)

  - horizontal_flip
    - 수평 방향으로 뒤집기를 한다.
    ![2017-3-8-CNN_Data_Augmentation_5_horizontal_flip](/assets/2017-3-8-CNN_Data_Augmentation_5_horizontal_flip.png)

  - vertical_flip
    - 수직 방향으로 뒤집기를 한다.
    ![2017-3-8-CNN_Data_Augmentation_5_vertical_flip](/assets/2017-3-8-CNN_Data_Augmentation_5_vertical_flip.png)

  - fill_mode
    - 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식

####ImageDataGenerator 디버깅
- Generator를 이용해 학습하기 전, 먼저 변형된 이미지에 이상한 점이 없는지 확인해본다.
- flow 함수를 사용

```python
import tensorflow as tf
import tf.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
            )
img = load_img(path) # PIL Image
x = img_to_array(img)
x = x[tf.newaxis, ...]

i = 0
for x_batch, y_batch in datagen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix=save_name, save_format='jpeg'):
  i += 1
  if i > 20:
    break
```

####ImageDataGenerator.flow
- 데이터와 라벨 배열을 가져와 augmented data를 배치 단위로 만든다.

- **parameter**
  - x (Input data)
    - Numpy array of rank 4
  - y (Labels)
  - batch_size (Int)
    - default : 32
  - shuffle (Boolean)
    - default : True
  - save_to_dir (None or str)
    - default : None
    - 증강된 데이터를 저장할 디렉터리 경로
  - save_prefix (str)
    - default : ''
    - 저장될 이미지의 파일 이름 지정
    - save_to_dir이 None이 아닐때
  - save_format
    - default : 'png'
    - 'png'와 'jpeg' 둘 중 하나 선택

- **example**
```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

####ImageDataGenerator.flow_from_directory
- **parameter**
  - directory (Str)
    - 목표 디렉터리의 경로 '/example/'
    - 클래스당 하나의 서브디렉토리를 포함되야한다.
  - target_size (Tuple of integers)
    - default : (256, 256)
  - color_mode
    - default : 'rgb'
    - 'grayscale', 'rgb', 'rgba' 중 하나
  - classes
    - default : None
    - 클래스 서브 디렉토리의 선택적 리스트 ex) ['dogs', 'cats']
    - 만약 None이면 입력된 디렉토리의 서브디렉토리 이름을 자동으로 채택한다.
  - class_mode
    - default : 'categorical'
    - 'categorical', 'binary', 'sparse', 'input', None 중 하나
    - 'categorical' : 2D 원-핫 인코딩 라벨
    - 'binary' : 1D 바이너리 라벨
    - 'sparse' : 1D integer 라벨
    - 'input' : 입력 이미지와 동일한 이미지
    - 'None' : 라벨 리턴 안함
  - batch_size
    - default : 32
  - shuffle
    - default : True
  - seed
    - 랜덤 시드
  - save_to_dir
    - flow와 동일
  - save_prefix
    - flow와 동일
  - save_format
    - flow와 동일

- **example_classification**
```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

- **example_segmentation**
```python
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```

##model.fit_generator
---
- **parameter**
  - train_generator
  - steps_per_epoch=nb_train_samples//batch_size
  - epochs=epochs
  - validation_data=validation_generator

##Segmentation Training Demo
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

seed = 1
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_generator = image_datagen.flow_from_directory(
    'example/segmentation/image',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'example/segmentation/mask',
    class_mode=None,
    seed=seed)

# case 1
# train_generator = zip(image_generator, mask_generator)
# model.fit_generator(train_generator,
#                    step_per_epoch=image_generator.samples // batch_size,
#                    epochs=50)

# case 2
epochs = 10
for e in range(epochs):
    print('Epochs', e)
    batches = 0
    for image, label in zip(next(image_generator), next(mask_generator)):
        print(image.shape, label.shape)
        batches += 1
        if batches >= image_generator.samples / 32:
            break
```
