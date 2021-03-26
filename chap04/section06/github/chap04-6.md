# chap 04-6 로지스틱 회귀 뉴런으로 단일층 신경망을 만듭니다

2021.03.26

<br>

### 01. 일반적인 신경망의 모습을 알아봅니다

![image01]

* 왼쪽이 입력층(input layer), 오른쪽이 출력층(output layer) 가운데 층은 은닉층(hidden layer)
* 단일층 신경망: 입력층과 출력층만 가지는 신경망

<br>

### 02. 단일층 신경망을 구현합니다

* 앞에서 구현한 LogisticNeuron 클래스 또한 단일층 신경망으로 몇 가지 기능을 추가하여 단일층 신경망 구현 (SingleLayer)

* 손실 함수의 결괏값 저장 기능 추가하기

  ```python
  class SingleLayer:
      
      def __init__(self):
      	self.w = None
      	self.b = None
          self.losses = []
          # ...
      
      def fit(self, x, y, epochs=100):
  	#	...
  
  	for i in index:		# 모든 샘플에 대해 반복
          z = self.forpass(x[i])		# 정방향 계산
          a = self.activation(z)		# 활성화 함수 적용
          err = -(y[i] - a)		# 오차 계산
          w_grad, b_grad = self.backprop(x[i], err)	# 역방향 계산
          self.w -= w_grad			# 가중치 업데이트
          self.b -= b_grad			# 절편 업데이트
          # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적합니다.
          a = np.clip(a, 1e-10, 1-1e-10)
          loss +- -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
          # 에포크마다 평균 손실을 저장합니다.
		self.losses.append(loss/len(y))
  ```
  
  * __init__() 메서드에 손실 함수의 결괏값을 저장할 리스트 self.losses를 만들고 손실 함수의 결괏값을 샘플 개수로 나눈 평균값을 저장
  * self.activation() 메서드로 계산한 a의 값을 np.clip() 함수로 조정 (a가 0에 가까워지면 손실값이 무한해지지 않기 위해서 조정)

<br>

### 03. 여러 가지 경사 하강법에 대해 알아봅니다

* 경사 하강법의 종류
  * 확률적 경사 하강법(stochastic gradient descent): 샘플 데이터 1개에 대한 그레디언트를 계산하는 방식
    * 계산 비용이 적은 대신 가중치가 최적값에 수렴하는 과정이 불안정함
  * 배치 경사 하강법(batch gradient descent): 전체 훈련 세트를 사용하여 한 번에 그레이디언트를 계산하는 방식
    * 가중치가 최적값에 수렴하는 과정은 안정적이지만 그만큼 계산 비용이 많이 듦
  * 미니 배치 경사 하강법(mini-batch gradient descent): 배치(batch) 크기를 작게 하여 훈련 세트를 여러 번 나누어 처리하는 방식
    * 확률적 경사 하강법과 배치 경사 하강법의 장점을 절충한 방식

* 매 에포크마다 훈련 세트의 샘플 순서를 섞어 사용하기

  * 경사하강법은 가중치 최적값을 제대로 찾기 위해 매 에포크마다 훈련 세트의 샘플 순서를 섞어 가중치의 최적값을 계산해야 함
  
  * 훈련 세트의 샘플 순서를 섞는 전형적인 방법: 넘파이 배열의 인덱스를 섞은 후 인덱스 순서대로 샘플을 뽑음
  
  * 코드로 구현 (np.random.permutation() 함수를 사용)
  
    ```python
    def fit(self, x, y, epochs=100):
        	self.w = np.ones(x.shape[1])	# 가중치를 초기화합니다. (1로 초기화)
        	self.b = 0						# 절편을 초기화합니다. (0으로 초기화)
        	for i in range(epochs):			# epochs만큼 반복합니다.
                loss = 0
                index = np.random.permutation(np.arange(len(x)))	# 인덱스를 섞습니다.
            	for i in indexes:				# 모든 샘플에 대해 반복합니다.
                    z = self.forpass(x[i])		# 정방향 계산
                    a = self.activation(z)		# 활성화 함수 적용
                    err = -(y[i] - a)			# 오차 계산
                    w_grad, b_grad = self.backprop(x[i], err)	#역방향 계산
                  self.w -= w_grad			# 가중치 업데이트
                    self.b -= b_grad			# 절편 업데이트
                    a = np.clip(a, 1e-10, 1-1e-10)	# 안전한 로그 계산을 위해 클리핑한 후 손실을 누적합니다.
                    loss += -(y[i]*np.log(a)+(1-y[i]*np.log(1-a)))	# 에포크마다 평균 손실을 저장합니다.
                self.losses.append(loss.len(y))
    ```
  
  * score() 메서드 추가하기 (정확도를 계산해 주는 메서드 추가)
  
    ```python
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]	# 정방향 계산
        return np.array(z) > 0					# 계단 함수 적용
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    ```
  
    * predict() 메서드에는 로지스틱 함수를 적용하지 않고 z 값의 크기만 비교하여 결과 반환
  
  * SingleLayer 클래스 구현
  
    ```python
    class SingleLayer:
        
        def __init__(self):
        	self.w = None
        	self.b = None
            self.losses = []
            
        def forpass(self, x):
            z = np.sum(x * self.w) + self.b
            return z
        
        def backprop(self, x, err):
            w_grad = x * err
            b_grad = 1 * err
            return w_grad, b_grad
        
        def fit(self, x, y, epochs=100):
        	self.w = np.ones(x.shape[1])	# 가중치를 초기화합니다. (1로 초기화)
        	self.b = 0						# 절편을 초기화합니다. (0으로 초기화)
        	for i in range(epochs):			# epochs만큼 반복합니다.
                loss = 0
                index = np.random.permutation(np.arange(len(x)))	# 인덱스를 섞습니다.
            	for i in indexes:				# 모든 샘플에 대해 반복합니다.
                    z = self.forpass(x[i])		# 정방향 계산
                    a = self.activation(z)		# 활성화 함수 적용
                    err = -(y[i] - a)			# 오차 계산
                    w_grad, b_grad = self.backprop(x[i], err)	#역방향 계산
                    self.w -= w_grad			# 가중치 업데이트
                    self.b -= b_grad			# 절편 업데이트
                    a = np.clip(a, 1e-10, 1-1e-10)	# 안전한 로그 계산을 위해 클리핑한 후 손실을 누적합니다.
                    loss += -(y[i]*np.log(a)+(1-y[i]*np.log(1-a)))	# 에포크마다 평균 손실을 저장합니다.
                self.losses.append(loss.len(y))
                
    	def predict(self, x):
        	z = [self.forpass(x_i) for x_i in x]	# 정방향 계산
        	return np.array(z) > 0					# 계단 함수 적용
    
    	def score(self, x, y):
        	return np.mean(self.predict(x) == y)
    ```

<br>

### 04. 단일층 신경망 훈련하기

1. 단일층 신경망 훈련하고 정확도 출력하기

   ```python
   layer = SingleLayer()
   layer.fit(x_train, y_train)
   layer.score(x_test, y_test)		# 0.9298245614035088
   ```

   * SingleLayer 객체를 만들고 훈련 세트(x_train, y_train)로 이 신경망을 훈련한 다음 score() 메서드로 정확도 출력
   * 에포크마다 훈련 세트를 무작위로 섞어 손실 함수의 값을 줄였기 때문에 성능이 좋아짐

2. 손실 함수 누적값 확인하기

   ```python
   plt.plot(layer.losses)
   plt.xlabel('epoch')
   plt.ylabel('loss')
   plt.show()
   ```

   ![image02]

   * 손실 함수의 결괏값으로 그래프를 그려보니 로지스틱 손실 함수의 값이 에포크가 진행됨에 따라 감소하고 있음을 확인할 수 있음