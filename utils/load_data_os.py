import os

# 현재 디렉터리 위치를 변경
# os.chdir('C:/')

# 현재 디렉터리 위치를 반환
# print(os.getcwd())

# 디렉터리 생성
# os.mkdir(디렉터리 이름)

# 디렉터리 삭제 (비어있을때만)
# os.rmdir(디렉터리 이름)

# 특정 경로에 존재하는 파일과 디렉터리 목록 반환
# print(os.listdir())

# 파일 혹은 디렉터리가 존재하는지 체크
# print(os.path.exists('.'))

# 디렉터리 여부 확인
# print(os.path.isdir('.'))

# 파일 여부 확인
# print(os.path.isfile('load_data_classification.py'))

# 해당 OS 형식에 맞도록 입력 받은 경로를 연결한다.
# os.path.join('C:/', 'test', 'test.py')
# >> 'C:\\test\\test.py

# 입력 받은 경로를 디렉터리 부분과 파일 부분으로 나눈다.
# os.path.join('C:\\test\\test.py)
# >> ('C:\\test', 'test.py')

# 입력 받은 경로를 확장자 부분과 그 외의 부분으로 나눈다.
# os.path.splittext('C:\\test\\test.py')
# >> ('C:\\test\\test', '.py')

# 특정 경로에 대해 정대 경로 얻기
# print(os.path.abspath('.'))



from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

model = VGG16(include_top=False,
              weights=None,
              input_shape=(228, 228, 3),
              classes=2)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

