# chap 08-4 합성곱 신경망을 만들고 훈련합니다

2021.04.01

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
      
      * 자동 미분의 사용 방법을 알아봅니다
        
        ```python
        x = tf.Variable(np.array([1.0, 2.0, 3.0]))
        with tf.GradientTape() as tape:
            y = x ** 3 + 2 * x + 5
        
        # 그래디언트를 계산합니다.
        print(tape.gradient(y, x))
        # tf.Tensor([ 5. 14. 29.], shape=(3,), dtype=float64)
        ```
        
        * 텐서플로와 같은 딥러닝 패키지들은 사용자가 작성한 연산을 계산 그래프(computationgraph)로 만들어 자동 미분 기능을 구현하고 이를 사용하면 임의의 파이썬 코드나 함수에 대한 미분값을 계산할 수 있음
      
      1. 역방향 계산 구현하기
      
         ```python
         def training(self, x, y):
                 m = len(x)                    # 샘플 개수를 저장합니다.
                 with tf.GradientTape() as tape:
                     z = self.forpass(x)       # 정방향 계산을 수행합니다.
                     # 손실을 계산합니다.
                     loss = tf.nn.softmax_cross_entropy_with_logits(y, z)
                     loss = tf.reduce_mean(loss)
                 ...
         ```
      
         * 자동 미분 기능을 사용하면 backprop() 메서드를 구현할 필요가 없으며 training() 메서드의 구성도 간단해짐
         * 정방향 계산을 수행한 다음 정방향 계산의 결과(z)와 타깃(y)을 기반으로 손실값을 계산함
      
      2. 그레이디언트 계산하기
      
         ```python
         def training(self, x, y):
             ...
             weights_list = [self.conv_w, self.conv_b, self.w1, self.b1, self.w2, self.b2]
             # 가중치에 대한 그래디언트를 계산합니다.
             grads = tape.gradient(loss, weights_list)
             # 가중치를 업데이트합니다.
             self.optimizer.apply_gradients(zip(grads, weights_list))
         ```
      
         * 가중치와 절편을 업데이트해야 하며 tape.gradient() 메서드를 사용하면 그레이디언트를 자동으로 계산할 수 있음
         * 합성곱층의 가중치와 절편인 con_w와 con_b를 포함하여 그레이디언트가 필요한 가중치를 리스트로 나열함

<br>

### 04. 옵티마이저 객체를 만들어 가중치 초기화하기

`training() 메서드에 등장하는 self.optimizer를 확률적 경사 하강법(SGD)을 사용하여 fit() 메서드에서 만들어 보겠습니다.`

1. fit() 메서드 수정하기

   ```python
   def fit(self, x, y, epochs=100, x_val=None, y_val=None):
           self.init_weights(x.shape, y.shape[1])    # 은닉층과 출력층의 가중치를 초기화합니다.
           self.optimizer = tf.optimizers.SGD(learning_rate=self.lr)
           # epochs만큼 반복합니다.
           for i in range(epochs):
               print('에포크', i, end=' ')
               # 제너레이터 함수에서 반환한 미니배치를 순환합니다.
               batch_losses = []
               for x_batch, y_batch in self.gen_batch(x, y):
                   print('.', end='')
                   self.training(x_batch, y_batch)
                   # 배치 손실을 기록합니다.
                   batch_losses.append(self.get_loss(x_batch, y_batch))
               print()
               # 배치 손실 평균내어 훈련 손실 값으로 저장합니다.
               self.losses.append(np.mean(batch_losses))
               # 검증 세트에 대한 손실을 계산합니다.
               self.val_losses.append(self.get_loss(x_val, y_val))
   ```

   * 옵티마이저 객체 생성 부분만 제외하면 MultiClassNetwork 클래스의 fit() 메서드와 거의 동일함
   * 텐서플로는 tf.optimizers 모듈 아래에 여러 종류의 경사 하강법을 구현해 놓았으며 SGD 옵티마이저 객체는 기본 경사 하강법임

2. init_weights() 메서드 수정하기

   ```python
   def init_weights(self, input_shape, n_classes):
           g = tf.initializers.glorot_uniform()
           self.conv_w = tf.Variable(g((3, 3, 1, self.n_kernels)))
           self.conv_b = tf.Variable(np.zeros(self.n_kernels), dtype=float)
           n_features = 14 * 14 * self.n_kernels
           self.w1 = tf.Variable(g((n_features, self.units)))          # (특성 개수, 은닉층의 크기)
           self.b1 = tf.Variable(np.zeros(self.units), dtype=float)    # 은닉층의 크기
           self.w2 = tf.Variable(g((self.units, n_classes)))           # (은닉층의 크기, 클래스 개수)
           self.b2 = tf.Variable(np.zeros(n_classes), dtype=float)     # 클래스 개수
   ```

   * 거중치를 glorot_uniform() 함수로 초기화한다는 점과 텐서플로의 자동 미분 기능을 사용하기 위해 가중치를 tf.Variable() 함수로 만든다는 점에서 큰 변화가 있음
   * 절편 변수를 가중치 변수와 동일하게 32비트 실수로 맞추기 위해 dtype 매개변수에 float을 지정

<br>

### 05. glorot_uniform()를 알아봅니다

<img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section4/image02.PNG?raw=true" alt="image02.PNG" style="zoom: 80%;" />

* 다음과 같은 손실 함수가 있을 때 경사 하강법은 출발점으로부터 기울기가 0이 되는 최저점을 찾아가는데 가중치를 적절하게 초기화하지 않으면 출발점이 적절하지 않은 곳에 설정되므로 왼쪽 모습과 같이 엉뚱한 곳에서 최적점이라는 판단을 내릴 수도 있음
* 지역 최적점(local minimum), 전역 최적점(global minimum)
* 글로럿 초기화 방식으로 가중치를 초기화합니다
  * 세이비어 글로럿(Xavier Glorot)이 제안하여 널리 사용되는 가중치 초기화 방식으로 균등하게 난수를 발생시켜 가중치를 초기화
  * glorot_uniform() 함수에서 만든 객체를 호출할 때 필요한 가중치 크기를 전달하여 글로럿 초기화를 사용함

<br>

* ConvolutionNetwork 클래시 전체 코드

  ```python
  class ConvolutionNetwork:
      
      def __init__(self, n_kernels=10, units=10, batch_size=32, learning_rate=0.1):
          self.n_kernels = n_kernels  # 합성곱의 커널 개수
          self.kernel_size = 3        # 커널 크기
          self.optimizer = None       # 옵티마이저
          self.conv_w = None          # 합성곱 층의 가중치
          self.conv_b = None          # 합성곱 층의 절편
          self.units = units          # 은닉층의 뉴런 개수
          self.batch_size = batch_size  # 배치 크기
          self.w1 = None              # 은닉층의 가중치
          self.b1 = None              # 은닉층의 절편
          self.w2 = None              # 출력층의 가중치
          self.b2 = None              # 출력층의 절편
          self.a1 = None              # 은닉층의 활성화 출력
          self.losses = []            # 훈련 손실
          self.val_losses = []        # 검증 손실
          self.lr = learning_rate     # 학습률
  
      def forpass(self, x):
          # 3x3 합성곱 연산을 수행합니다.
          c_out = tf.nn.conv2d(x, self.conv_w, strides=1, padding='SAME') + self.conv_b
          # 렐루 활성화 함수를 적용합니다.
          r_out = tf.nn.relu(c_out)
          # 2x2 최대 풀링을 적용합니다.
          p_out = tf.nn.max_pool2d(r_out, ksize=2, strides=2, padding='VALID')
          # 첫 번째 배치 차원을 제외하고 출력을 일렬로 펼칩니다.
          f_out = tf.reshape(p_out, [x.shape[0], -1])
          z1 = tf.matmul(f_out, self.w1) + self.b1     # 첫 번째 층의 선형 식을 계산합니다
          a1 = tf.nn.relu(z1)                          # 활성화 함수를 적용합니다
          z2 = tf.matmul(a1, self.w2) + self.b2        # 두 번째 층의 선형 식을 계산합니다.
          return z2
      
      def init_weights(self, input_shape, n_classes):
          g = tf.initializers.glorot_uniform()
          self.conv_w = tf.Variable(g((3, 3, 1, self.n_kernels)))
          self.conv_b = tf.Variable(np.zeros(self.n_kernels), dtype=float)
          n_features = 14 * 14 * self.n_kernels
          self.w1 = tf.Variable(g((n_features, self.units)))          # (특성 개수, 은닉층의 크기)
          self.b1 = tf.Variable(np.zeros(self.units), dtype=float)    # 은닉층의 크기
          self.w2 = tf.Variable(g((self.units, n_classes)))           # (은닉층의 크기, 클래스 개수)
          self.b2 = tf.Variable(np.zeros(n_classes), dtype=float)     # 클래스 개수
          
      def fit(self, x, y, epochs=100, x_val=None, y_val=None):
          self.init_weights(x.shape, y.shape[1])    # 은닉층과 출력층의 가중치를 초기화합니다.
          self.optimizer = tf.optimizers.SGD(learning_rate=self.lr)
          # epochs만큼 반복합니다.
          for i in range(epochs):
              print('에포크', i, end=' ')
              # 제너레이터 함수에서 반환한 미니배치를 순환합니다.
              batch_losses = []
              for x_batch, y_batch in self.gen_batch(x, y):
                  print('.', end='')
                  self.training(x_batch, y_batch)
                  # 배치 손실을 기록합니다.
                  batch_losses.append(self.get_loss(x_batch, y_batch))
              print()
              # 배치 손실 평균내어 훈련 손실 값으로 저장합니다.
              self.losses.append(np.mean(batch_losses))
              # 검증 세트에 대한 손실을 계산합니다.
              self.val_losses.append(self.get_loss(x_val, y_val))
  
      # 미니배치 제너레이터 함수
      def gen_batch(self, x, y):
          bins = len(x) // self.batch_size                   # 미니배치 횟수
          indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞습니다.
          x = x[indexes]
          y = y[indexes]
          for i in range(bins):
              start = self.batch_size * i
              end = self.batch_size * (i + 1)
              yield x[start:end], y[start:end]   # batch_size만큼 슬라이싱하여 반환합니다.
              
      def training(self, x, y):
          m = len(x)                    # 샘플 개수를 저장합니다.
          with tf.GradientTape() as tape:
              z = self.forpass(x)       # 정방향 계산을 수행합니다.
              # 손실을 계산합니다.
              loss = tf.nn.softmax_cross_entropy_with_logits(y, z)
              loss = tf.reduce_mean(loss)
  
          weights_list = [self.conv_w, self.conv_b,
                          self.w1, self.b1, self.w2, self.b2]
          # 가중치에 대한 그래디언트를 계산합니다.
          grads = tape.gradient(loss, weights_list)
          # 가중치를 업데이트합니다.
          self.optimizer.apply_gradients(zip(grads, weights_list))
     
      def predict(self, x):
          z = self.forpass(x)                 # 정방향 계산을 수행합니다.
          return np.argmax(z.numpy(), axis=1) # 가장 큰 값의 인덱스를 반환합니다.
      
      def score(self, x, y):
          # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환합니다.
          return np.mean(self.predict(x) == np.argmax(y, axis=1))
  
      def get_loss(self, x, y):
          z = self.forpass(x)                 # 정방향 계산을 수행합니다.
          # 손실을 계산하여 저장합니다.
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z))
          return loss.numpy()
  ```

<br>

### 06. 합성곱 신경망 훈련하기

`직접 만든 합성곱 신경망을 사용하여 합성곱 신경망 모델을 만들고 패션 MNIST 데이터 세트를 훈련시켜 보겠습니다.`

1. 데이터 세트 불러오기

   ```python
   (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
   # 텐서플로를 사용해 패션 MNIST 데이터 세트를 불러옴
   ```

2. 훈련 데이터 세트를 훈련 세트와 검증 세트로 나누기

   ```python
   from sklearn.model_selection import train_test_split
   x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)
   # 사이킷런을 사용하여 훈련 데이터 세트를 훈련 세트와 검증 세트로 나눔
   ```

3. 타깃을 원-핫 인코딩으로 변환하기

   ```python
   y_train_encoded = tf.keras.utils.to_categorical(y_train)
   y_val_encoded = tf.keras.utils.to_categorical(y_val)
   ```

4. 입력 데이터 준비하기

   ```python
   x_train = x_train.reshape(-1, 28, 28, 1)
   x_val = x_val.reshape(-1, 28, 28, 1)
   x_train.shape	# (48000, 28, 28, 1)
   ```

   * 합성곱 신경망은 입력 데이터(이미지)를 일렬로 펼칠 필요가 없이 높이와 너비 차원을 그대로 유지한 채 신경망에 주입함
   * 마지막 컬러 채널을 추가해야 하므로 넘파이 reshape() 메서드를 사용하여 마지막 차원을 간단히 추가함

5. 입력 데이터 표준화 전처리하기

   ```python
   x_train = x_train / 255
   x_val = x_val / 255
   # 입력 데이터를 0~1 사이의 값으로 조정함
   ```

6. 모델 훈련하기

   ```python
   cn = ConvolutionNetwork(n_kernels=10, units=100, batch_size=128, learning_rate=0.01)
   cn.fit(x_train, y_train_encoded, x_val=x_val, y_val=y_val_encoded, epochs=20)
   ```

   * 합성곱 커널 10개, 완전 연결층의 뉴런 100개를 사용하고 배치 크기는 128개, 학습률은 0.01로 지정
   * 모델을 20번의 에포크 동안 훈련

7. 훈련, 검증 손실 그래프 그리고 검증 세트의 정확도 확인하기

   * 훈련 손실과 검증 손실 그래프

     ```python
     plt.plot(cn.losses)
     plt.plot(cn.val_losses)
     plt.ylabel('loss')
     plt.xlabel('iteration')
     plt.legend(['train_loss', 'val_loss'])
     plt.show()
     ```

     <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section4/image03.PNG?raw=true" alt="image03.PNG"  />

   * 모델의 검증 세트에 대한 정확도 측정

     ```python
     cn.score(x_val, y_val_encoded)	# 0.8806666666666667
     ```

     