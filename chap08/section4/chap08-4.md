# chap 08-4 합성곱 신경망을 만들고 훈련합니다

2021.04.

`텐서플로가 제공하는 합성곱 함수와 자동 미분 기능을 사용하여 합성곱 신경망을 구현해 보겠습니다.`

<br>

### 01. 합성곱 신경망의 전체 구조를 한 번 더 살펴보세요

<img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section4/image01.PNG?raw=true" alt="image01.PNG" style="zoom:67%;" />

* 28 * 28 크기의 흑백 이미지와 3 * 3 크기의 커널 10개로 합성곱을 수행하고 2 * 2 크기 최대 풀링을 수행하여 특성 맵의 크기를 줄임
* 특성 맵을 일렬로 펼쳐서 100개의 뉴런을 가진 완전 연결층과 연결시키고 10개의 클래스를 구분하기 위해 소프트맥스 함수를 연결

<br>

### 02. 합성곱 신경망의 정방향 계산 구현하기

`이번에 구현할 합성곱 신경망 클래스는 ConvolutionNetwork로 코드 구성은 이전에 구현한 클래스들과 대체로 비슷하지만 합성곱과 렐루 함수 그리고 풀링이 적용된다는 점이 다름`

1. 합성곱 적용하기

   ```python
   def forpass(self, x):
           # 3x3 합성곱 연산을 수행합니다.
           c_out = tf.nn.conv2d(x, self.conv_w, strides=1, padding='SAME') + self.conv_b
   ```

   * conv2d() 함수를 통해 합성곱을 수행한 다음 절편 self.conv_d를 더해야 함
   * 크기가 10인 1차원 배열 self.conv_b는 자동으로 conv2d() 함수의 결과 마지막 차원에 브로드캐스팅 됨
   * 합섭곱을 수행하는 conv2d() 함수에 전달한 매개변수 값
     * self.conv_w: 합성곱에 사용할 가중치로 3 * 3 * 1 크기의 커널을 10개 사용하므로 가중치의 전체 크기는 3 * 3 * 1* 10
     * stride, padding: 특성 맵의 가로와 세로 크기를 일정하게 만들기 위하여 stride는 1, padding은 'SAME'으로 지정

2. 렐루 함수 적용하기

   ```python
   def forpass(self, x):
       ...
       # 렐루 활성화 함수를 적용합니다.
       r_out = tf.nn.relu(c_out)
       ...
   ```

3. 풀링 적용하고 완전 연결층 수정하기

   ```python
   def forpass(self, x):
       ...
       # 2x2 최대 풀링을 적용합니다.
       p_out = tf.nn.max_pool2d(r_out, ksize=2, strides=2, padding='VALID')
       # 첫 번째 배치 차원을 제외하고 출력을 일렬로 펼칩니다.
       f_out = tf.reshape(p_out, [x.shape[0], -1])
       z1 = tf.matmul(f_out, self.w1) + self.b1     # 첫 번째 층의 선형 식을 계산합니다
       a1 = tf.nn.relu(z1)                          # 활성화 함수를 적용합니다
       z2 = tf.matmul(a1, self.w2) + self.b2        # 두 번째 층의 선형 식을 계산합니다.
       return z2
   ```

   * max_pool2d() 함수를 사용하여 2 * 2 크기의 풀링을 적용하여 특성 맵의 크기를 줄이고 tf.reshape() 함수를 사용해 일렬로 펼침
   * np.dot() 함수를 텐서플로의 tf.matmul() 함수로 바꾸고 완전 연결층의 활성화 함수도 시그모이드 함수 대신 렐루 함수 사용

<br>

### 03. 합성곱 신경망의 역방향 계산 구현하기

* 그레이디언트를 구하기 위해 역방향 계산을 직접 구현하는 대신 텐서플로의 자동 미분(automatic differentiation) 기능을 사용

* 자동 미분의 사용 방법을 알아봅니다

  * 텐서플로와 같은 딥러닝 패키지들은 사용자가 작성한 연산을 계산 그래프(computation graph)로 만들어 자동 미분 기능을 구현

  * 사용자가 작성한 연산을 바탕으로 자동 미분하여 미분값을 얻어낸 예

    ```python
    x = tf.Variable(np.array([1.0, 2.0, 3.0]))
    with tf.GradientTape() as tape:
        y = tf.nn.softmax(x)
    
    # 그래디언트를 계산합니다.
    print(tape.gradient(y, x))
    # tf.Tensor([9.99540153e-18 2.71703183e-17 7.38565826e-17], shape=(3,), dtype=float64)
    ```

    * 텐서플로의 자동 미분 기능을 사용하려면 with 블럭으로 tf.GradientTape() 객체가 감시할 코드를 감싸야 함
    * tape 객체는 with 블럭 안에서 일어나는 모든 연산을 기록하고 텐서플로 변수인 tf.Variable 객체를 자동으로 추적함
    * 그레이디언트를 계산하려면 미분 대상 객체와 변수를 tape 객체의 gradient() 메서드에 전달해야 함
    * 합성곱 신경망의 역방향 계산 구현과 그레이디언트 계산하기
      1. 역방향 계산 구현하기
      2. 그레이디언트 계산하기

<br>

### 04. 옵티마이저 객체를 만들어 가중치 초기화하기

1. fit() 메서드 수정하기
2. init_weights() 메서드 수정하기

<br>

### 05. glorot_uniform()를 알아봅니다

<br>

### 06. 합성곱 신경망 훈련하기

1. 데이터 세트 불러오기
2. 훈련 데이터 세트를 훈련 세트와 검증 세트로 나누기
3. 타깃을 원-핫 인코딩으로 변환하기
4. 입력 데이터 준비하기
5. 입력 데이터 표준화 전처리하기
6. 모델 훈련하기
7. 훈련, 검증 손실 그래프 그리고 검증 세트의 정확도 확인하기