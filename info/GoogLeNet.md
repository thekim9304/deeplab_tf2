# GoogLeNet
---
#### v1:Going deeper with convolutions
---
![](https://miro.medium.com/max/5176/1*ZFPOSAted10TPd3hBQU8iQ.png)

- 파라미터를 줄이기 위해서 FC layer가 없음 : global average pooling으로 대체

**Inception module**
![](https://norman3.github.io/papers/images/google_inception/f01.png)
- 연산량을 줄이기 위해서 (b)와 같이 1x1 conv를 추가해 feature map의 depth를 줄이고 계산함

**auxiliary classifier**
  - Vanishing Gradient 문제를 해결하기 위한 방법
  - 네트워크 중간 중간에 softmax 결과를 뽑는다
  - 대신 지나치게 영향을 주는 것을 막기 위해 auxiliary classifier의 loss에는 0.3을 곱해준다.
  - 학습 과정에서만 사용하고 테스트에서는 사용하지 않는다.

---
#### v2:

**3x3 conv**
![](https://norman3.github.io/papers/images/google_inception/f07.png)

- v1에서 사용하던 5x5 conv도 부담스러워서 3x3 conv 두 개로 대체

**factorization**
<img src=https://norman3.github.io/papers/images/google_inception/f09.png width=60%>

- N x N의 형태로 수행하던 Conv를 1 x N과 N x 1로 인수분해 하는 기법 : 연산량이 33% 줄어든다.

**auxiliary classifier**
- v1에서는 앞에 두 개를 추가했지만 v2에서는 하나만 사용



---
[참조](https://norman3.github.io/papers/docs/google_inception.html)
