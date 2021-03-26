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

* 매 에포크마다 훈련 세트의 샘플 순서를 섞어 사용하기

  ```python
  def fit(self, x, y, epochs=100):
      	self.w = np.ones(x.shape[1])	# 가중치를 초기화합니다. (1로 초기화)
      	self.b = 0						# 절편을 초기화합니다. (0으로 초기화)
      	for i in range(epochs):			# epochs만큼 반복합니다.
              loss = 0
              index = np.random.permutation(np.arange(len(x)))
          	for i in indexes:
                  z = self.forpass(x[i])
                  a = self.activation(z)
                  err = -(y[i] - a)
                  w_grad, b_grad = self.backprop(x[i], err)
                  self.w -= w_grad
                  self.b -= b_grad
                  a = np.clip(a, 1e-10, 1-1e-10)
                  loss += -(y[i]*np.log(a)+(1-y[i]*np.log(1-a)))
              self.losses.append(loss.len(y))
  ```

  



<br>

### 04. 단일층 신경망 훈련하기