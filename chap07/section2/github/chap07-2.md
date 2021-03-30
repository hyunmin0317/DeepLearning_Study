# chap 07-2 텐서플로와 케라스를 사용하여 신경망을 만듭니다

2021.03.

<br>

### 01. 케라스에 대해 알아봅니다

* 0

  ```python
  # 훈련할 가중치 변수를 선언합니다.
  w = tf.Variable(tf.zeros(shape=(1)))
  b = tf.Variable(tf.zeros(shape=(1)))
  
  # 경사 하강법 옵티마이저를 설정합니다.
  optimizer =tf.optimizers.SGD(lr = 0.01)
  # 에포크 횟수만큼 훈련합니다.
  num_epochs = 10
  for step in range(num_epochs):
      
      # 자동 미분을 위해 연산 과정을 기록합니다.
      with tf.GradientTape() as tape:
          z_net = w * x_train + b
          z_net = tf.reshape(z_net, [-1])
          sqr_errors = tf.square(y_train - z_net)
          mean_cost = tf.reduce_mean(sqr_errors)
      # 손실 함수에 대한 가중치의 그레이디언트를 계산합니다.
      grads = tape.gradient(mean_cost, [w, b])
      # 옵티마이저에 그레이디언트를 반영합니다.
      optimizer.apply_gradients(zip(grads, [w, b]))
  ```

  

* 0

<br>

### 02. Sequential 클래스의 사용 방법을 알아봅니다

<br>

### 03. Dense 클래스의 사용 방법을 알아봅니다

<br>

### 04. 모델의 최적화 알고리즘과 손실 함수를 설정합니다

<br>

### 05. 모델을 훈련하고 예측합니다

<br>

### 06. 케라스로 다중 분류 신경망을 만들어봅니다
