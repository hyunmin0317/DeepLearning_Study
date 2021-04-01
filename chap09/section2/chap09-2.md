# chap 09-2 순환 신경망을 만들고 텍스트를 분류합니다

2021.04.02

<br>

### 01. 훈련 세트와 검증 세트를 준비합니다

1. 텐서플로에서 IMDB 데이터 세트 불러오기

   ```python
   import numpy as np
   from tensorflow.keras.datasets import imdb
   
   (x_train_all, y_train_all), (x_test, y_test) = imdb.load_data(skip_top=20, num_words=100)
   ```

2. 훈련 세트의 크기 확인

   ```python
   print(x_train_all.shape, y_train_all.shape)	# (25000,) (25000,)
   ```

3. 훈련 세트의 샘플 확인하기

   ```python
   print(x_train_all[0])
   # [2, 2, 22, 2, 43, 2, 2, 2, 2, 65, 2, 2, 66, 2, 2, 2, 36, 2, 2, 25, 2, 43, 2, 2, 50, 2, 2, 2, 35, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 39, 2, 2, 2, 2, 2, 2, 38, 2, 2, 2, 2, 50, 2, 2, 2, 2, 2, 2, 22, 2, 2, 2, 2, 2, 22, 71, 87, 2, 2, 43, 2, 38, 76, 2, 2, 2, 2, 22, 2, 2, 2, 2, 2, 2, 2, 2, 2, 62, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 66, 2, 33, 2, 2, 2, 2, 38, 2, 2, 25, 2, 51, 36, 2, 48, 25, 2, 33, 2, 22, 2, 2, 28, 77, 52, 2, 2, 2, 2, 82, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 36, 71, 43, 2, 2, 26, 2, 2, 46, 2, 2, 2, 2, 2, 2, 88, 2, 2, 2, 2, 98, 32, 2, 56, 26, 2, 2, 2, 2, 2, 2, 2, 22, 21, 2, 2, 26, 2, 2, 2, 30, 2, 2, 51, 36, 28, 2, 92, 25, 2, 2, 2, 65, 2, 38, 2, 88, 2, 2, 2, 2, 2, 2, 2, 2, 32, 2, 2, 2, 2, 2, 32]
   ```

4. 훈련 세트에서 2 제외하기

   ```python
   for i in range(len(x_train_all)):
       x_train_all[i] = [w for w in x_train_all[i] if w > 2]
   
   print(x_train_all[0])
   # [22, 43, 65, 66, 36, 25, 43, 50, 35, 39, 38, 50, 22, 22, 71, 87, 43, 38, 76, 22, 62, 66, 33, 38, 25, 51, 36, 48, 25, 33, 22, 28, 77, 52, 82, 36, 71, 43, 26, 46, 88, 98, 32, 56, 26, 22, 21, 26, 30, 51, 36, 28, 92, 25, 65, 38, 88, 32, 32]
   ```

5. 어휘 사전 내려받기

   ```python
   word_to_index = imdb.get_word_index()
   word_to_index['movie']	# 17
   ```

   <br>

6. 훈련 세트의 정수를 영단어로 변환하기

   ```python
   index_to_word = {word_to_index[k]: k for k in word_to_index}
   
   for w in x_train_all[0]:
       print(index_to_word[w - 3], end=' ')
   # film just story really they you just there an from so there film film were great just so much film would really at so you what they if you at film have been good also they were just are out because them all up are film but are be what they have don't you story so because all all 
   ```

7. 훈련 샘플의 길이 확인하기

   ```python
   print(len(x_train_all[0]), len(x_train_all[1]))	#59 32
   ```

8. 훈련 세트의 타깃 데이터 확인하기

   ```python
   print(y_train_all[:10])	# [1 0 0 1 0 0 1 0 1 0]
   ```

9. 검증 세트를 준비합니다

   ```python
   np.random.seed(42)
   random_index = np.random.permutation(25000)
   
   x_train = x_train_all[random_index[:20000]]
   y_train = y_train_all[random_index[:20000]]
   x_val = x_train_all[random_index[20000:]]
   y_val = y_train_all[random_index[20000:]]
   ```

<br>

### 02. 샘플의 길이 맞추기

<img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section2/image01.PNG?raw=true" alt="image01.PNG" style="zoom:80%;" />

1. 텐서플로로 샘플의 길이 맞추기

   ```python
   from tensorflow.keras.preprocessing import sequence
   
   maxlen=100
   x_train_seq = sequence.pad_sequences(x_train, maxlen=maxlen)
   x_val_seq = sequence.pad_sequences(x_val, maxlen=maxlen)
   ```

2. 길이를 조정한 훈련 세트의 크기와 샘플 확인하기

   ```python
   print(x_train_seq.shape, x_val_seq.shape)	# (20000, 100) (5000, 100)
   ```

   * 샘플 길이를 변경한 훈련 세트의 첫번째 샘플 확인

     ```python
     print(x_train_seq[0])
     # [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     #   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     #   0  0  0  0  0  0  0  0  0  0  0  0  0  0 35 40 27 28 40 22 83 31 85 45
     #  24 23 31 70 31 76 30 98 32 22 28 51 75 56 30 33 97 53 38 46 53 74 31 35
     #  23 34 22 58]
     ```

<br>

### 03. 샘플을 원-핫 인코딩하기

1. 텐서플로로 원-핫 인코딩하기

   ```python
   from tensorflow.keras.utils import to_categorical
   
   x_train_onehot = to_categorical(x_train_seq)
   x_val_onehot = to_categorical(x_val_seq)
   ```

2. 원-핫 인코딩으로 변환된 변수 x_train_onehot의 크기를 확인

   ```python
   print(x_train_onehot.shape)	# (20000, 100, 100)
   ```

3. x_train_onehot의 크기 확인

   ```python
   print(x_train_onehot.nbytes)	# 800000000
   ```

<br>

### 04. 순환 신경망 클래스 구현하기

1. `__init__()` 메서드 수정하기

   ```python
   def __init__(self, n_cells=10, batch_size=32, learning_rate=0.1):
           self.n_cells = n_cells     # 셀 개수
           self.batch_size = batch_size     # 배치 크기
           self.w1h = None            # 은닉 상태에 대한 가중치
           self.w1x = None            # 입력에 대한 가중치
           self.b1 = None             # 순환층의 절편
           self.w2 = None             # 출력층의 가중치
           self.b2 = None             # 출력층의 절편
           self.h = None              # 순환층의 활성화 출력
           self.losses = []           # 훈련 손실
           self.val_losses = []       # 검증 손실
           self.lr = learning_rate    # 학습률
   ```

2. 직교 행렬 방식으로 가중치 초기화하기

   ```python
   def init_weights(self, n_features, n_classes):
           orth_init = tf.initializers.Orthogonal()
           glorot_init = tf.initializers.GlorotUniform()
           
           self.w1h = orth_init((self.n_cells, self.n_cells)).numpy() # (셀 개수, 셀 개수)
           self.w1x = glorot_init((n_features, self.n_cells)).numpy() # (특성 개수, 셀 개수)
           self.b1 = np.zeros(self.n_cells)                           # 은닉층의 크기
           self.w2 = glorot_init((self.n_cells, n_classes)).numpy()   # (셀 개수, 클래스 개수)
           self.b2 = np.zeros(n_classes)
   ```

3. 정방향 계산 구현하기

   ```python
   def forpass(self, x):
       self.h = [np.zeros((x.shape[0], self.n_cells))]   # 은닉 상태를 초기화합니다.
       ...
   ```

4. 입력 x의 첫 번째 배치 차원과 두 번째 타임 스텝 차원을 바꿈

   ```python
   	...
   	# 배치 차원과 타임 스텝 차원을 바꿉니다.
   	seq = np.swapaxes(x, 0, 1)
       ...
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section2/image02.PNG?raw=true" alt="image02.PNG" style="zoom: 67%;" />

   <br>

5. 샘플의 모든 타임 스텝에 대한 정방향 계산을 수행함

   ```python
   	...
       # 순환 층의 선형 식을 계산합니다.
       for x in seq:
           z1 = np.dot(x, self.w1x) + np.dot(self.h[-1], self.w1h) + self.b1
           h = np.tanh(z1)                    # 활성화 함수를 적용합니다.
           self.h.append(h)                   # 역전파를 위해 은닉 상태 저장합니다.
           z2 = np.dot(h, self.w2) + self.b2  # 출력층의 선형 식을 계산합니다.
   	return z2
   ```

6. 역방향 계산 구현하기

   ```python
   def backprop(self, x, err):
           m = len(x)       # 샘플 개수
           
           # 출력층의 가중치와 절편에 대한 그래디언트를 계산합니다.
           w2_grad = np.dot(self.h[-1].T, err) / m
           b2_grad = np.sum(err) / m
           # 배치 차원과 타임 스텝 차원을 바꿉니다.
           seq = np.swapaxes(x, 0, 1)
           
           w1h_grad = w1x_grad = b1_grad = 0
           # 셀 직전까지 그래디언트를 계산합니다.
           err_to_cell = np.dot(err, self.w2.T) * (1 - self.h[-1] ** 2)
           # 모든 타임 스텝을 거슬러가면서 그래디언트를 전파합니다.
           for x, h in zip(seq[::-1][:10], self.h[:-1][::-1][:10]):
               w1h_grad += np.dot(h.T, err_to_cell)
               w1x_grad += np.dot(x.T, err_to_cell)
               b1_grad += np.sum(err_to_cell, axis=0)
               # 이전 타임 스텝의 셀 직전까지 그래디언트를 계산합니다.
               err_to_cell = np.dot(err_to_cell, self.w1h) * (1 - h ** 2)
           
           w1h_grad /= m
           w1x_grad /= m
           b1_grad /= m
       
           return w1h_grad, w1x_grad, b1_grad, w2_grad, b2_grad
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section2/image03.PNG?raw=true" alt="image03.PNG" style="zoom: 80%;" />

7. 나머지 메서드 수정하기

   ```python
   class RecurrentNetwork:
       
       def __init__(self, n_cells=10, batch_size=32, learning_rate=0.1):
           self.n_cells = n_cells     # 셀 개수
           self.batch_size = batch_size     # 배치 크기
           self.w1h = None            # 은닉 상태에 대한 가중치
           self.w1x = None            # 입력에 대한 가중치
           self.b1 = None             # 순환층의 절편
           self.w2 = None             # 출력층의 가중치
           self.b2 = None             # 출력층의 절편
           self.h = None              # 순환층의 활성화 출력
           self.losses = []           # 훈련 손실
           self.val_losses = []       # 검증 손실
           self.lr = learning_rate    # 학습률
   
       def forpass(self, x):
           self.h = [np.zeros((x.shape[0], self.n_cells))]   # 은닉 상태를 초기화합니다.
           # 배치 차원과 타임 스텝 차원을 바꿉니다.
           seq = np.swapaxes(x, 0, 1)
           # 순환 층의 선형 식을 계산합니다.
           for x in seq:
               z1 = np.dot(x, self.w1x) + np.dot(self.h[-1], self.w1h) + self.b1
               h = np.tanh(z1)                    # 활성화 함수를 적용합니다.
               self.h.append(h)                   # 역전파를 위해 은닉 상태 저장합니다.
               z2 = np.dot(h, self.w2) + self.b2  # 출력층의 선형 식을 계산합니다.
           return z2
   
       def backprop(self, x, err):
           m = len(x)       # 샘플 개수
           
           # 출력층의 가중치와 절편에 대한 그래디언트를 계산합니다.
           w2_grad = np.dot(self.h[-1].T, err) / m
           b2_grad = np.sum(err) / m
           # 배치 차원과 타임 스텝 차원을 바꿉니다.
           seq = np.swapaxes(x, 0, 1)
           
           w1h_grad = w1x_grad = b1_grad = 0
           # 셀 직전까지 그래디언트를 계산합니다.
           err_to_cell = np.dot(err, self.w2.T) * (1 - self.h[-1] ** 2)
           # 모든 타임 스텝을 거슬러가면서 그래디언트를 전파합니다.
           for x, h in zip(seq[::-1][:10], self.h[:-1][::-1][:10]):
               w1h_grad += np.dot(h.T, err_to_cell)
               w1x_grad += np.dot(x.T, err_to_cell)
               b1_grad += np.sum(err_to_cell, axis=0)
               # 이전 타임 스텝의 셀 직전까지 그래디언트를 계산합니다.
               err_to_cell = np.dot(err_to_cell, self.w1h) * (1 - h ** 2)
           
           w1h_grad /= m
           w1x_grad /= m
           b1_grad /= m
       
           return w1h_grad, w1x_grad, b1_grad, w2_grad, b2_grad
       
       def sigmoid(self, z):
           z = np.clip(z, -100, None)            # 안전한 np.exp() 계산을 위해
           a = 1 / (1 + np.exp(-z))              # 시그모이드 계산
           return a
       
       def init_weights(self, n_features, n_classes):
           orth_init = tf.initializers.Orthogonal()
           glorot_init = tf.initializers.GlorotUniform()
           
           self.w1h = orth_init((self.n_cells, self.n_cells)).numpy() # (셀 개수, 셀 개수)
           self.w1x = glorot_init((n_features, self.n_cells)).numpy() # (특성 개수, 셀 개수)
           self.b1 = np.zeros(self.n_cells)                           # 은닉층의 크기
           self.w2 = glorot_init((self.n_cells, n_classes)).numpy()   # (셀 개수, 클래스 개수)
           self.b2 = np.zeros(n_classes)
           
       def fit(self, x, y, epochs=100, x_val=None, y_val=None):
           y = y.reshape(-1, 1)
           y_val = y_val.reshape(-1, 1)
           np.random.seed(42)
           self.init_weights(x.shape[2], y.shape[1])    # 은닉층과 출력층의 가중치를 초기화합니다.
           # epochs만큼 반복합니다.
           for i in range(epochs):
               print('에포크', i, end=' ')
               # 제너레이터 함수에서 반환한 미니배치를 순환합니다.
               batch_losses = []
               for x_batch, y_batch in self.gen_batch(x, y):
                   print('.', end='')
                   a = self.training(x_batch, y_batch)
                   # 안전한 로그 계산을 위해 클리핑합니다.
                   a = np.clip(a, 1e-10, 1-1e-10)
                   # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
                   loss = np.mean(-(y_batch*np.log(a) + (1-y_batch)*np.log(1-a)))
                   batch_losses.append(loss)
               print()
               self.losses.append(np.mean(batch_losses))
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
           a = self.sigmoid(z)       # 활성화 함수를 적용합니다.
           err = -(y - a)            # 오차를 계산합니다.
           # 오차를 역전파하여 그래디언트를 계산합니다.
           w1h_grad, w1x_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)
           # 셀의 가중치와 절편을 업데이트합니다.
           self.w1h -= self.lr * w1h_grad
           self.w1x -= self.lr * w1x_grad
           self.b1 -= self.lr * b1_grad
           # 출력층의 가중치와 절편을 업데이트합니다.
           self.w2 -= self.lr * w2_grad
           self.b2 -= self.lr * b2_grad
           return a
      
       def predict(self, x):
           z = self.forpass(x)          # 정방향 계산을 수행합니다.
           return z > 0                 # 스텝 함수를 적용합니다.
       
       def score(self, x, y):
           # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환합니다.
           return np.mean(self.predict(x) == y.reshape(-1, 1))
   
       def update_val_loss(self, x_val, y_val):
           z = self.forpass(x_val)            # 정방향 계산을 수행합니다.
           a = self.sigmoid(z)                # 활성화 함수를 적용합니다.
           a = np.clip(a, 1e-10, 1-1e-10)     # 출력 값을 클리핑합니다.
           val_loss = np.mean(-(y_val*np.log(a) + (1-y_val)*np.log(1-a)))
           self.val_losses.append(val_loss)
   ```

<br>

### 05. 순환 신경망 모델 훈련시키기

1. 순환 신경망 모델 훈련시키기

   ```python
   rn = RecurrentNetwork(n_cells=32, batch_size=32, learning_rate=0.01)
   
   rn.fit(x_train_onehot, y_train, epochs=20, x_val=x_val_onehot, y_val=y_val)
   ```

2. 훈련, 검증 세트에 대한 손실 그래프 그리기

   ```python
   import matplotlib.pyplot as plt
   
   plt.plot(rn.losses)
   plt.plot(rn.val_losses)
   plt.show()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section2/image04.PNG?raw=true" alt="image04.PNG"  />

3. 검증 세트 정확도 평가하기

   ```python
   rn.score(x_val_onehot, y_val)	# 0.6572
   ```

   