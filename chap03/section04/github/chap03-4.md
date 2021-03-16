# chap 03-4 선형 회귀를 위한 뉴런을 만듭니다

2021.03.17

```python
앞에서 만든 경사 하강법 알고리즘을 Neuron이라는 이름의 파이썬 클래스로 만들기
```

<br>

### 01. Neuron 클래스 만들기

* Neuron 클래스의 전체 구조

  ```python
  class Neuron:
      def __init__(self):
          # 초기화 작업을 수행합니다.
          # 필요한 메서드를 추가합니다.
  ```

  * 최근에는 뉴런이라는 명칭 대신 '유닛(unit)' 이라는 명칭을 즐겨 사용함

* Neuron 클래스 구현
  1. __init__() 메서드 작성하기
  
     * __init__() 메서드에 필요한 변수 선언 (가중치 w와 절편 b의 시작값을 1.0으로 지정)
  
       ```python
       def __init__(self):
           self.w = 1.0
           self.b = 1.0
       ```
  
  2. 정방향 계산 만들기
  
     * 정방향 계산: 뉴런으로 도식화한 상태에서 ŷ을 구하는 방향을 보고 만든 용어, ŷ = wx + b
  
       ```python
       def forpass(self, x):
           y_hat = x * self.w + self.b
           return y_hat
       ```
  
  3. 역방향 계산 만들기
  
     * 역방향 계산: ŷ을 계산하여 y와의 오차(역전파)를 계산하여 w와 b의 그레이디언트를 계산하는 과정
  
     * 가중치와 절편을 업데이트하여 점차 훈련 데이터에 최적화된 가중치와 절편을 구함
  
       ```python
       def backprop(self, x, err):
           w_grad = x * err
           b_grad = 1 * err
           return w_grad, b_grad
       ```
  
  4. 지금까지 작성한 Neuron 클래스(딥러닝이 사용하는 경사 하강법 알고리즘의 핵심이 담겨 있음)
  
     ```python
     class Neuron:
         
         def __init__(self):
         self.w = 1.0
         self.b = 1.0
         
         def forpass(self, x):
         y_hat = x * self.w + self.b
         return y_hat
     
     	def backprop(self, x, err):
         w_grad = x * err
         b_grad = 1 * err
         return w_grad, b_grad
     ```
  
     
  
  5. 훈련을 위한 fit() 메서드 구현하기(훈련 데이터를 통해 가중치와 절편을 업데이트하는 메서드-훈련)
  
     ```python
     def fit(self, x, y, epochs=100):
         for i in range(epochs):				# 에포크만큼 반복합니다.
             for x_i, y_i in zip(x, y):		# 모든 샘플에 대해 반복합니다.
                 y_hat = self.forpass(x_i)	# 정방향 계산
                 err = -(y_i - y_hat)		# 오차 계산
                 w_grad, b_grad = self.backprop(x_i, err)	# 역방향 계산
                 self.w -= w_grad			# 가중치 업데이트
                 self.b -= b_grad			# 절편 업데이트
     ```
  
  6. 모델 훈련하기(학습시키기)
  
     ```python
     neuron = Neuron()
     neuron.fit(x, y)
     ```
  
     
  
  7. 학습이 완료된 모델의 가중치와 절편 확인하기