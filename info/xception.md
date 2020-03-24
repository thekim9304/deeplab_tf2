# Xception
#### Xception: Deep Learning with Depthwise Separable Convolutions
---
- **Depthwise separable convolution**
  - 일반적으로 `separable convolution`으로 불린다.
  - 원래는 depthwise convolution 뒤에 pointwise convolution이 뒤따르는 형태
  - Xception에서는 조금 수정된 버전을 사용한다.
  ![](https://miro.medium.com/max/844/1*J8dborzVBRBupJfvR7YhuA.png)


  - **Depthwise convolution**
  ![](https://blogfiles.pstatic.net/MjAxOTAxMDNfMTc3/MDAxNTQ2NDk0OTQxMjQ0.nHWjwkwaiwKmscpuIDy5qllZtFdDZzmBbd8t6NNjN28g.LUVIBXc8q1c4f2fruMURBcn_Ds6xHEWd3hT_rqoC6jMg.PNG.worb1605/image.png)
    - H * W * C의 feature map을 채널 단위로 분리하여 각각 conv filter를 적용한 후 합친다.
    - conv filter에 훨씬 적응 파라미터를 가지고 동일한 크기의 출력을 낼 수 있다.

  - **Pointwise convolution**
    - 1x1 conv filter 사용
    - 공간 방향의 convolution은 하지 않고, 채널 방향의 convolution만 한다.
    - 특징 맵의 차원을 늘리거나 줄일 때 사용

- **Inception에서 separable convolution**
  ```python
  x = Conv2D(32, (1, 1), use_bias=False)(x) # pointwise conv
  x = DepthwiseConv2D((3, 3), activation='relu')(x) # depthwise conv
  ```
  - 선행하는 layer에 bias를 사용하지 않는 이유가 있다.




[참고](https://sike6054.github.io/blog/paper/fifth-post/)
