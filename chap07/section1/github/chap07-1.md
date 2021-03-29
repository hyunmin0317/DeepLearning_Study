# chap 07-1 여러 개의 이미지를 분류하는 다층 신경망을 만듭니다

2021.03.30

<br>

### 01. 다중 분류 신경망을 알아봅니다

![image01.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image01.PNG?raw=true)

* 다중 분류는 마지막 출력층에 여러 개의 뉴런을 놓아 신경망을 구성함
* 이진 분류는 양성 클래스에 대한 확률 하나만 출력하고 다중 분류는 각 클래스에 대한 확률값을 출력
* 다중 분류 신경망은 출력층에 분류할 클래스 개수만큼 뉴런을 배치함

<br>

### 02. 다중 분류의 문제점과 소프트맥스 함수를 알아봅니다

![image02.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image02.PNG?raw=true)

* 왼쪽 출력층의 활성화값은 [0.9, 0.8, 0.7], 오른쪽 출력층의 활성화값은 [0.5, 0.2, 0.1]로 비교해 보면 미묘한 차이가 있음

* 활성화 출력의 합이 1이 아니면 비교하기 어렵습니다
  
  * 출력층의 활성화
  
* 소프트맥스 함수 적용해 출력 강도를 정규화합니다

  ![image03.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image03.PNG?raw=true)

  * 소프트맥스 함수(softmax function): 출력 강도 정규화하는 함수 (전체 출력값의 합을 1로 만듦)

    ![image04.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image04.PNG?raw=true)

<br>

### 03. 크로스 엔트로피 손실 함수를 도입합니다

![image05.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image05.PNG?raw=true)

* 크로스 엔트로피 손실 함수의 시그마 기호 위의 값(c)은 전체 클래스 개수를 의미함
* 크로스 엔트로피 손실 함수는 로지스틱 손실 함수의 일반화 버전으로 로지스틱 함수가 크로스 엔트로피 함수의 이진 분류 버전

<br>

### 04. 크로스 엔트로피 손실 함수를 미분합니다

* z1에 대하여 미분합니다

  ![image06.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image06.PNG?raw=true)

* 손실 함수 L을 z1에 대하여 미분하기 위해 연쇄 법칙을 따라 차례로 미분

  ![image07.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image07.PNG?raw=true)

  * 손실 함수 L을 a1, a2, a3에 대하여 미분

    ![image08.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image08.PNG?raw=true)

  * a1을 z1에 대해 미분

    ![image09.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image09.PNG?raw=true)

  * a2를 z1에 대해 미분

    ![image10.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image10.PNG?raw=true)

  * a3를 z1에 대해 미분

    ![image11.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image11.PNG?raw=true)

  * 크로스 엔트로피 손실 함수의 미분 결과

    ![image12.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image12.PNG?raw=true)

* 로지스틱 손실 함수의 미분과 일치하여 크로스 엔트로피 손실 함수를 역전파에 사용하기 위해 코드로 따로 구현할 필요가 없음

<br>

### 05. 다중 분류 신경망을 구현합니다

1. 소프트맥스 함수 추가하기

   ```python
   def sigmoid(self, z):
           z = np.clip(z, -100, None)            # 안전한 np.exp() 계산을 위해
           a = 1 / (1 + np.exp(-z))              # 시그모이드 계산
           return a
       
       def softmax(self, z):
           # 소프트맥스 함수
           z = np.clip(z, -100, None)            # 안전한 np.exp() 계산을 위해
           exp_z = np.exp(z)
           return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)
   ```

   * np.exp()와 np.sum()을 사용하여 소프트맥스 함수 구현

2. 정방향 계산하기

   ```python
   def forpass(self, x):
           ...
           self.a1 = self.sigmoid(z1)
           ...
   ```

   * activation() 메서드의 이름을 sigmoid()로 바꿨으므로 이를 수정, 나머지 코드는 동일

3. 가중치 초기화하기

   ```python
    def init_weights(self, n_features, n_classes):
           ...
           self.w2 = np.random.normal(0, 1, 
                                      (self.units, n_classes))   # (은닉층의 크기, 클래스 개수)
           self.b2 = np.zeros(n_classes)
   ```

   * w2를 random.normal() 함수를 이용하여 배열의 각 원소의 값 정규 분포를 따르는 무작위 수로 초기화
   * b2는 모두 0으로 초기화

4. fit 메서드 수정하기

   ```python
   def fit(self, x, y, epochs=100, x_val=None, y_val=None):
           np.random.seed(42)
           self.init_weights(x.shape[1], y.shape[1])    # 은닉층과 출력층의 가중치를 초기화합니다.
           # epochs만큼 반복합니다.
           for i in range(epochs):
               loss = 0
               print('.', end='')
               # 제너레이터 함수에서 반환한 미니배치를 순환합니다.
               for x_batch, y_batch in self.gen_batch(x, y):
                   a = self.training(x_batch, y_batch)
                   # 안전한 로그 계산을 위해 클리핑합니다.
                   a = np.clip(a, 1e-10, 1-1e-10)
                   # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
                   loss += np.sum(-y_batch*np.log(a))
               self.losses.append((loss + self.reg_loss()) / len(x))
               # 검증 세트에 대한 손실을 계산합니다.
               self.update_val_loss(x_val, y_val)
   ```

   * 가중치를 초기화하는 init_weights() 메서드를 호출할 때 클래스의 개수를 매개변수의 값으로 넘겨주는 부분을 수정

   <br>

5. training() 메서드 수정하기

   ```python
   def training(self, x, y):
           m = len(x)                # 샘플 개수를 저장합니다.
           z = self.forpass(x)       # 정방향 계산을 수행합니다.
           a = self.softmax(z)       # 활성화 함수를 적용합니다.
   ```

   * 출력층의 활성화 함수를 activation() 메서드에서 softmax() 메서드로 바꿈

6. predict() 메서드 수정하기

   ```python
   def predict(self, x):
           z = self.forpass(x)          # 정방향 계산을 수행합니다.
           return np.argmax(z, axis=1)  # 가장 큰 값의 인덱스를 반환합니다.
   ```

   * 정방향 계산에서 얻은 출력 중 가장 큰 값의 인덱스를 구하고 이 값이 예측 클래스가 됨

7. score() 메서드 수정하기

   ```python
   def score(self, x, y):
           # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환합니다.
           return np.mean(self.predict(x) == np.argmax(y, axis=1))
   ```

   * predict() 메서드의 결과와 타깃 y의 클래스를 비교하고 이를 위해 배열 y의 행을 따라 가장 큰 값의 인덱스를 구해 사용

8. 검증 손실 계산하기

   ```python
   def update_val_loss(self, x_val, y_val):
           ...
           a = self.softmax(z)                # 활성화 함수를 적용합니다.
           ...
           # 크로스 엔트로피 손실과 규제 손실을 더하여 리스트에 추가합니다.
           val_loss = np.sum(-y_val*np.log(a))
           ...
   ```

   * update_val_loss() 메서드에서 사용하는 활성화 함수를 softmax()로 로지스틱 손실 계산을 크로스 엔트로피 손실 계산으로 바꿈

<br>

* MultiClassNetwork 클래스의 전체 코드

  ```python
  class MultiClassNetwork:
      
      def __init__(self, units=10, batch_size=32, learning_rate=0.1, l1=0, l2=0):
          self.units = units         # 은닉층의 뉴런 개수
          self.batch_size = batch_size     # 배치 크기
          self.w1 = None             # 은닉층의 가중치
          self.b1 = None             # 은닉층의 절편
          self.w2 = None             # 출력층의 가중치
          self.b2 = None             # 출력층의 절편
          self.a1 = None             # 은닉층의 활성화 출력
          self.losses = []           # 훈련 손실
          self.val_losses = []       # 검증 손실
          self.lr = learning_rate    # 학습률
          self.l1 = l1               # L1 손실 하이퍼파라미터
          self.l2 = l2               # L2 손실 하이퍼파라미터
  
      def forpass(self, x):
          z1 = np.dot(x, self.w1) + self.b1        # 첫 번째 층의 선형 식을 계산합니다
          self.a1 = self.sigmoid(z1)               # 활성화 함수를 적용합니다
          z2 = np.dot(self.a1, self.w2) + self.b2  # 두 번째 층의 선형 식을 계산합니다.
          return z2
  
      def backprop(self, x, err):
          m = len(x)       # 샘플 개수
          # 출력층의 가중치와 절편에 대한 그래디언트를 계산합니다.
          w2_grad = np.dot(self.a1.T, err) / m
          b2_grad = np.sum(err) / m
          # 시그모이드 함수까지 그래디언트를 계산합니다.
          err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1 - self.a1)
          # 은닉층의 가중치와 절편에 대한 그래디언트를 계산합니다.
          w1_grad = np.dot(x.T, err_to_hidden) / m
          b1_grad = np.sum(err_to_hidden, axis=0) / m
          return w1_grad, b1_grad, w2_grad, b2_grad
      
      def sigmoid(self, z):
          z = np.clip(z, -100, None)            # 안전한 np.exp() 계산을 위해
          a = 1 / (1 + np.exp(-z))              # 시그모이드 계산
          return a
      
      def softmax(self, z):
          # 소프트맥스 함수
          z = np.clip(z, -100, None)            # 안전한 np.exp() 계산을 위해
          exp_z = np.exp(z)
          return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)
   
      def init_weights(self, n_features, n_classes):
          self.w1 = np.random.normal(0, 1, 
                                     (n_features, self.units))  # (특성 개수, 은닉층의 크기)
          self.b1 = np.zeros(self.units)                        # 은닉층의 크기
          self.w2 = np.random.normal(0, 1, 
                                     (self.units, n_classes))   # (은닉층의 크기, 클래스 개수)
          self.b2 = np.zeros(n_classes)
          
      def fit(self, x, y, epochs=100, x_val=None, y_val=None):
          np.random.seed(42)
          self.init_weights(x.shape[1], y.shape[1])    # 은닉층과 출력층의 가중치를 초기화합니다.
          # epochs만큼 반복합니다.
          for i in range(epochs):
              loss = 0
              print('.', end='')
              # 제너레이터 함수에서 반환한 미니배치를 순환합니다.
              for x_batch, y_batch in self.gen_batch(x, y):
                  a = self.training(x_batch, y_batch)
                  # 안전한 로그 계산을 위해 클리핑합니다.
                  a = np.clip(a, 1e-10, 1-1e-10)
                  # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
                  loss += np.sum(-y_batch*np.log(a))
              self.losses.append((loss + self.reg_loss()) / len(x))
              # 검증 세트에 대한 손실을 계산합니다.
              self.update_val_loss(x_val, y_val)
  
      # 미니배치 제너레이터 함수
      def gen_batch(self, x, y):
          length = len(x)
          bins = length // self.batch_size # 미니배치 횟수
          if length % self.batch_size:
              bins += 1                    # 나누어 떨어지지 않을 때
          indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞습니다.
          x = x[indexes]
          y = y[indexes]
          for i in range(bins):
              start = self.batch_size * i
              end = self.batch_size * (i + 1)
              yield x[start:end], y[start:end]   # batch_size만큼 슬라이싱하여 반환합니다.
              
      def training(self, x, y):
          m = len(x)                # 샘플 개수를 저장합니다.
          z = self.forpass(x)       # 정방향 계산을 수행합니다.
          a = self.softmax(z)       # 활성화 함수를 적용합니다.
          err = -(y - a)            # 오차를 계산합니다.
          # 오차를 역전파하여 그래디언트를 계산합니다.
          w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)
          # 그래디언트에서 페널티 항의 미분 값을 뺍니다
          w1_grad += (self.l1 * np.sign(self.w1) + self.l2 * self.w1) / m
          w2_grad += (self.l1 * np.sign(self.w2) + self.l2 * self.w2) / m
          # 은닉층의 가중치와 절편을 업데이트합니다.
          self.w1 -= self.lr * w1_grad
          self.b1 -= self.lr * b1_grad
          # 출력층의 가중치와 절편을 업데이트합니다.
          self.w2 -= self.lr * w2_grad
          self.b2 -= self.lr * b2_grad
          return a
     
      def predict(self, x):
          z = self.forpass(x)          # 정방향 계산을 수행합니다.
          return np.argmax(z, axis=1)  # 가장 큰 값의 인덱스를 반환합니다.
      
      def score(self, x, y):
          # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환합니다.
          return np.mean(self.predict(x) == np.argmax(y, axis=1))
  
      def reg_loss(self):
          # 은닉층과 출력층의 가중치에 규제를 적용합니다.
          return self.l1 * (np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + \
                 self.l2 / 2 * (np.sum(self.w1**2) + np.sum(self.w2**2))
  
      def update_val_loss(self, x_val, y_val):
          z = self.forpass(x_val)            # 정방향 계산을 수행합니다.
          a = self.softmax(z)                # 활성화 함수를 적용합니다.
          a = np.clip(a, 1e-10, 1-1e-10)     # 출력 값을 클리핑합니다.
          # 크로스 엔트로피 손실과 규제 손실을 더하여 리스트에 추가합니다.
          val_loss = np.sum(-y_val*np.log(a))
          self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))
  ```

<br>

### 06. 의류 이미지를 분류합니다

* MultiClassNetwork 클래스를 사용하여 '의류 이미지 분류하기'라는 다중 분류 문제 해결

* 패션 MNIST 데이터 세트 사용 - [패션 MNIST 데이터 세트 깃허브](https://github.com/zalandoresearch/fashion-mnist)

* 코랩에 텐서플로 최신 버전을 설치

  `!pip install tensorflow_gpu==2.0.0`

<br>

### 07. 의류 데이터를 준비합니다

1. 텐서플로 임포트하기

   ```python
   import tensorflow as tf
   ```

2. 텐서플로 버전 확인하기

   ```python
   tf.__version__	# '2.4.1'
   ```

3. 패션 MNIST 데이터 세트 불러오기

   ```python
   (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
   ```

   * 텐서플로의 keras.datasets.fashion_mnist.load_data() 함수를 사용하여 패션 MNIST 데이터 세트를 불러옴
   * load_data() 함수는 입력과 타깃을 하나의 튜플로 묶어 훈련 세트와 테스트 세트를 반환하는 함수로 이를 변수 4개에 나누어 담음

4. 훈련 세트의 크기 확인하기

   ```python
   print(x_train_all.shape, y_train_all.shape)	# (60000, 28, 28) (60000,)
   ```

   

5. imshow() 함수로 샘플 이미지 확인하기

   ```python
   import matplotlib.pyplot as plt
   plt.imshow(x_train_all[0], cmap='gray')
   plt.show()
   ```

   ![image13.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image13.PNG?raw=true)

   * imshow() 함수는 넘파이 배열을 입력 받아 이미지를 그리고 cmap 매개변수로 색을 표현하는 값인 컬러맵(colormap) 설정
   * cmap 매개변수의 기본 설정은 짙은 녹색과 밝은 노란색 사이인 'viridis'이며 패션 MNIST는 흑백 이미지이므로 'gray'로 지정

   <br>

6. 타깃의 내용과 의미 확인하기

   ```python
   print(y_train_all[:10])	# [9 0 0 3 0 2 7 2 5 5]
   class_names = ['티셔츠/윗도리', '바지', '스웨터', '드레스', '코트', '샌들', '셔츠', '스니커즈', '가방', '앵클부츠']
   print(class_names[y_train_all[0]])	# 앵클부츠
   ```

   

7. 타깃 분포 확인하기

   ```python
   np.bincount(y_train_all)	# array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000])
   ```

   

   <br>

8. 훈련 세트와 검증 세트 고르게 나누기

   ```python
   from sklearn.model_selection import train_test_split
   x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)
   np.bincount(y_train)	# array([4800, 4800, 4800, 4800, 4800, 4800, 4800, 4800, 4800, 4800])
   np.bincount(y_val)	# array([1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200])
   ```

   

9. 입력 데이터 정규화하기

   ```python
   x_train = x_train / 255
   x_val = x_val / 255
   ```

   

10. 훈련 세트와 검증 세트의 차원 변경하기

    ```python
    x_train = x_train.reshape(-1, 784)
    x_val = x_val.reshape(-1, 784)
    print(x_train.shape, x_val.shape)	# (48000, 784) (12000, 784)
    ```

    

<br>

### 08. 타깃 데이터를 준비하고 다중 분류 신경망을 훈련합니다

1. 타깃을 원-핫 인코딩으로 변환하기

2. 배열의 각 원소를 뉴런의 출력값과 비교하기

3. to_categorical 함수 사용해 원-핫 인코딩하기

   ```python
   tf.keras.utils.to_categorical([0, 1, 3])
   # array([[1., 0., 0., 0.],
   #       [0., 1., 0., 0.],
   #       [0., 0., 0., 1.]], dtype=float32)
   y_train_encoded = tf.keras.utils.to_categorical(y_train)
   y_val_encoded = tf.keras.utils.to_categorical(y_val)
   print(y_train_encoded.shape, y_val_encoded.shape)	# (48000, 10) (12000, 10)
   print(y_train[0], y_train_encoded[0])	# 6 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
   ```

   

4. MultiClassNetwork 클래스로 다중 분류 신경망 훈련하기

   ```python
   fc = MultiClassNetwork(units=100, batch_size=256)
   fc.fit(x_train, y_train_encoded, x_val=x_val, y_val=y_val_encoded, epochs=40)
   # ........................................
   ```

   

5. 훈련 손실, 검증 손실 그래프와 훈련 모델 점수 확인하기

   ```python
   plt.plot(fc.losses)
   plt.plot(fc.val_losses)
   plt.ylabel('loss')
   plt.xlabel('iteration')
   plt.legend(['train_loss', 'val_loss'])
   plt.show()
   fc.score(x_val, y_val_encoded)	# 0.8150833333333334
   ```

   ![image14.PNG](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section1/github/image14.PNG?raw=true)