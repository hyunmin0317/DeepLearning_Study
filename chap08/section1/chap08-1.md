# chap 08-1 합성곱 연산에 대해 알아봅니다

2021.03.31

`합성곱 신경망을 이해하려면 먼저 합성곱(convolution) 연산과 교차 상관(cross-correlation) 연산에 대해 알아야 함`

<br>

### 01. 합성곱을 그림으로 이해합니다

<img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image01.PNG?raw=true" alt="image01.PNG" style="zoom:67%;" />

* 배열 하나 선택해 뒤집기
  * 두 배열 x와 w가 있다고 가정하고 두 배열 중 원소수가 적은 배열 w의 원소 순서를 뒤집어보면 그 배열을 w^r이라고 표현
* 첫 번째 합성곱
  * 뒤집은 배열을 배열 x의 왼쪽 끝자리에 맞춰 놓은 다음 각 배열 원소끼리 곱한 후 더함 (점 곱 연산) - 63
* 두 번째 합성곱
  * 뒤집은 배열을 오른쪽으로 한 칸 이동하여 각 배열 원소끼리 곱한 후 더함 - 48
* 나머지 합성곱
  * 같은 방식으로 뒤집은 배열을 오른쪽으로 한 칸씩 이동하여 x의 끝에 도착할 때까지 합성곱을 수행함
* 합성곱은 수식으로 x * w와 같이 표기

<br>

### 02. 합성곱 구현하기

1. 넘파이 배열 정의하고 배열 하나 선택해 뒤집기

   * 넘파이 배열로 w와 x 정의

     ```python
     import numpy as np
     w = np.array([2, 1, 5, 3])
     x = np.array([2, 8, 3, 7, 1, 2, 0, 4, 5])
     ```

   * w 배열 뒤집기

     ```python
     w_r = np.flip(w)
     # w_r = w[::-1]	파이썬의 슬라이스 연산자
     print(w_r)	# [3 5 1 2]
     ```

     * 넘파이의 flip() 함수나 파이썬의 슬라이스 연산자를 통해 배열을 뒤집을 수 있음

2. 넘파이의 점 곱으로 합성곱 수행하기

   ```python
   for i in range(6):
       print(np.dot(x[i:i+4], w_r))
   ```

   * x 배열을 한 칸씩 이동하면서 넘파이의 점 곱을 이용하여 합성곱을 수행

3. 싸이파이로 합성곱 수행하기

   ```python
   from scipy.signal import convolve
   convolve(x, w, mode='valid')	# array([63, 48, 49, 28, 21, 20])
   ```

   * 싸이파이는 합성곱을 위한 함수 convolve()를 제공

<br>

### 03. 합성곱 신경망은 진짜 합성곱을 사용하지 않습니다

`대부분의 딥러닝 패키지들은 합성곱 신경망을 만들 때 합성곱이 아니라 교차 상관을 사용함`

* 합성곱과 교차 상관은 아주 비슷합니다

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image02.PNG?raw=true" alt="image02.PNG" style="zoom:67%;" />

  * 교차 상관은 합성곱과 동일한 방법으로 연산이 진행되지만 `미끄러지는 배열을 뒤집지 않는다`는 점이 다름

  * 싸이파이의 correlate() 함수로 구현

    ```python
    from scipy.signal import correlate
    correlate(x, w, mode='valid')	# array([48, 57, 24, 25, 16, 39])
    ```

* 합성곱 신경망에서 교차 상관을 사용하는 이유를 알아봅니다

  * 모델을 훈련하기 전에 가중치 배열의 요소들을 무작위로 초기화하므로 가중치를 뒤집을 이유가 없음
  * 모델 훈련 과정 간단 정리
    * 가중치를 무작위 값으로 초기화합니다.
    * 모든 샘플에 대하여 정방향과 역방향 계산을 수행하여 가중치를 조금씩 학습(업데이트)합니다.

<br>

### 04. 패딩과 스트라이드를 이해합니다

* 패딩(padding): 원본 배열의 양 끝에 빈 원소를 추가하는 것을 말함

* 스트라이드(stride): 미끄러지는 배열의 간격을 조절하는 것을 말함

* 패딩의 종류

  * 밸리드 패딩(valid padding) - 원본 배열의 원소가 합성곱 연산에 참여하는 정도가 서로 다릅니다

    <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image03.PNG?raw=true" alt="image03.PNG" style="zoom:67%;" />

    * 원본 배열에 패딩을 추가하지 않고 미끄러지는 배열이 원본 배열의 끝으로 갈 때까지 교차 상관을 수행함
    * 밸리드 패딩의 결과로 얻는 배열의 크기는 원본 배열보다 항상 작으며 원본 배열의 각 원소가 연산에 참여하는 정도가 다름

  <br>

  * 풀 패딩(full padding) - 원본 배열 원소의 연산 참여도를 동일하게 만듭니다

    <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image04.PNG?raw=true" alt="image04.PNG" style="zoom:67%;" />

    * 원본 배열의 모든 요소가 동일하게 연산에 참여하는 패딩 방식
    * 제로 패딩(zero padding): 원본 배열의 양 끝에 가상의 원소 0을 추가하여 원본 배열의 원소가 연산에 동일하게 참여하게 함

  * 세임 패딩(same padding) - 출력 배열의 길이를 원본 배열의 길이와 동일하게 만듭니다 

    <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image05.PNG?raw=true" alt="image05.PNG" style="zoom:67%;" />

    * 출력 배열의 길이가 원본 배열의 길이와 같아지도록 원본 배열에 제로 패딩을 추가
    * 합성곱 신경망에서는 대부분 세임 패딩을 사용함

<br>

* 스트라이드 - 미끄러지는 간격을 조정합니다

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image06.PNG?raw=true" alt="image06.PNG" style="zoom:67%;" />

  * 미끄러지는 배열의 간격으로 보통 1로 지정함

<br>

### 05. 2차원 배열에서 합성곱을 수행합니다

<img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image07.PNG?raw=true" alt="image07.PNG" style="zoom:67%;" />

* 1차원 배열의 합성곱과 비슷하게 수행되며 합성곱의 수행 방향은 원본 배열의 왼쪽에서 오른쪽으로, 위에서 아래쪽으로 1칸씩 이동하며 배열 원소끼리 곱함

* 싸이파이의 correlated2d() 함수를 사용하여 2차원 배열의 합성곱을 계산

  ```python
  x = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
  w = np.array([[2, 0], [0, 0]])
  from scipy.signal import correlate2d
  correlate2d(x, w, mode='valid')
  # array([[ 2,  4],
  #        [ 8, 10]])
  ```

* 세임 패딩을 적용하는 방법

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image08.PNG?raw=true" alt="image08.PNG" style="zoom:67%;" />

  ```python
  correlate2d(x, w, mode='same')
  # array([[ 2,  4,  6],
  #        [ 8, 10, 12],
  #        [14, 16, 18]])
  ```

* 스트라이드를 2로 지정한 경우

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image09.PNG?raw=true" alt="image09.PNG" style="zoom:67%;" />

<br>

### 06. 텐서플로로 합성곱을 수행합니다

`텐서플로의 합성곱을 위한 함수도 싸이파이와 동일한 결과를 출력하는지 확인 - 원본 배열(입력), 미끄러지는 배열(가중치)`

* 합성곱 신경망의 입력은 일반적으로 4차원 배열입니다

  * conv2d() 함수: 텐서플로에서 2차원 합성곱을 수행하는 함수로 입력으로 4차원 배열을 기대함

  * 입력 배열의 구성

    <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image10.PNG?raw=true" alt="image10.PNG" style="zoom:67%;" />

  * 입력과 가중치에 세임 패딩을 적용하여 합성곱 수행

    <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section1/image11.PNG?raw=true" alt="image11.PNG" style="zoom:67%;" />

* 2차원 배열을 4차원 배열로 바꿔 합성곱을 수행합니다

  * 입력 배열을 reshape() 메서드로 2차원 배열에서 4차원 배열로 바꾸고 astype() 메서드로 입력의 자료형을 실수로 바꿈

    ```python
    import tensorflow as tf
    x_4d = x.astype(np.float).reshape(1, 3, 3, 1)
    w_4d = w.reshape(2, 2, 1, 1)
    ```

  * 스트라이드는 1, 패딩은 세임 페딩을 적용

    ```python
    c_out = tf.nn.conv2d(x_4d, w_4d, strides=1, padding='SAME')
    ```

  * (3, 3) 크기로 변환하여 출력

    ```python
    c_out.numpy().reshape(3, 3)
    # array([[ 2.,  4.,  6.],
    #        [ 8., 10., 12.],
    #        [14., 16., 18.]])
    ```

<br>

### 07. 패션 MNIST 데이터 세트를 합성곱 신경망에 적용하면 어떻게 될까요?

* 패션 MNIST 데이터 세트를 합성곱 신경망에서는 28 * 28 입력을 펼치지 않고 그대로 사용하여 3 * 3, 5 * 5 가중치로 합성곱 적용
* 거중치 배열의 크기는 훨씬 작아졌고 입력의 특징을 더 잘 찾기 때문에 합성곱 신경망이 이미지 분류에서 성능이 뛰어남

<br>

### 08. 이 책에서는 합성곱의 가중치를 필터 또는 커널이라고 부릅니다

* 합성곱의 가중치를 필터(filter) 또는 커널(kernel)이라고 부름
* 합성곱의 필터 1개를 지칭할 때는 '커널'이라고 필터 전체를 지칭할 때는 일반 신경망과 동일하게 '가중치' 라고 하겠습니다.