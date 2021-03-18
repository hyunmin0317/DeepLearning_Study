# chap 03-4 선형 회귀를 위한 뉴런을 만듭니다

2021.03.18

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


<br>

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
       * 가중치와 절편은 인스턴스 변수 w와 b에 저장되어 있는 값을 사용

  3. 역방향 계산 만들기

     * 역방향 계산: ŷ을 계산하여 y와의 오차(역전파)를 계산하여 w와 b의 그레이디언트를 계산하는 과정

       ```python
       def backprop(self, x, err):
           w_grad = x * err	# 가중치에 대한 그레디언트를 계산
           b_grad = 1 * err	# 절편에 대한 그레디언트를 계산
           return w_grad, b_grad
       ```

       * 가중치의 그레디언트는 x와 오차(err)를 곱하고 절편의 그레디언트는 1과 오차(err)를 곱해서 구함
       * 가중치와 절편을 업데이트하여 점차 훈련 데이터에 최적화된 가중치와 절편을 구함

  4. 지금까지 작성한 Neuron 클래스(딥러닝이 사용하는 경사 하강법 알고리즘의 핵심이 담겨 있음)

     ```python
     class Neuron:
         
         def __init__(self):
         	self.w = 1.0	# 가중치 초기화
         	self.b = 1.0	# 절편 초기화
         
         def forpass(self, x):
         	y_hat = x * self.w + self.b	# 직선 방정식 계산
         	return y_hat
     
     	def backprop(self, x, err):
         	w_grad = x * err	# 가중치에 대한 그레이디언트 계산
         	b_grad = 1 * err	# 절편에 대한 그레이디언트 계산
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

     * forpass() 메서드를 호출하여 ŷ을 구하고 오차(err)를 계산하고 backprop() 메서드를 호출하여 가중치와 절편에 대한 그레디언트를 구한 뒤 그레이디언트를 가중치와 절편에서 빼서 가중치와 절편 업데이트
     * 이 과정을 모든 훈련 샘플에 대해 수행하고(1 에포크) 적절한 가중치와 절편이 구해질 만큼(100 에포크) 반복

  6. 모델 훈련하기(학습시키기)

     ```python
     neuron = Neuron()
     neuron.fit(x, y)
     ```

     * Neuron 클래스의 객체 neuron을 생성하고 fit() 메서드에 입력 데이터(x)와 타깃 데이터(y) 전달하여 모델 훈련

  7. 학습이 완료된 모델의 가중치와 절편 확인하기

     * 학습이 완료된 가중치(neuron.w)와 절편(neuron.b)을 이용하여 산점도 위에 직선 그래프 그리기

     ```python
     plt.scatter(x, y)
     pt1 = (-0.1, -0.1 * neuron.w + neuron.b)
     pt2 = (0.15, 0.15 * neuron.w + neuron.b)
     plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
     plt.xlabel('x')
     plt.ylabel('y')
     plt.show()
     ```

     ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap03/section04/github/image01.PNG?raw=true)<br>
     
     <br>
  
* Neuron 클래스

  ```python
  class Neuron:
      
      def __init__(self):
      	self.w = 1.0	# 가중치 초기화
      	self.b = 1.0	# 절편 초기화
      
      def forpass(self, x):
      	y_hat = x * self.w + self.b	# 직선 방정식 계산
      	return y_hat
  
  	def backprop(self, x, err):
      	w_grad = x * err	# 가중치에 대한 그레이디언트 계산
      	b_grad = 1 * err	# 절편에 대한 그레이디언트 계산
      	return w_grad, b_grad
  
  def fit(self, x, y, epochs=100):
      for i in range(epochs):				# 에포크만큼 반복합니다.
          for x_i, y_i in zip(x, y):		# 모든 샘플에 대해 반복합니다.
              y_hat = self.forpass(x_i)	# 정방향 계산
              err = -(y_i - y_hat)		# 오차 계산
              w_grad, b_grad = self.backprop(x_i, err)	# 역방향 계산
              self.w -= w_grad			# 가중치 업데이트
              self.b -= b_grad			# 절편 업데이트
  ```

  