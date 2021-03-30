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

  ```python
  # 신경망 모델을 만듭니다.
  model = tf.keras.models.Sequential()
  # 완전 연결층을 추가합니다. 
  model.add(tf.keras.layers.Dense(1))
  # 옵티마이저와 손실 함수를 지정합니다.
  model.compile(optimizer='sgd', loss='mse')
  # 훈련 데이터를 사용하여 에포크 횟수만큼 훈련합니다.
  model.fit(x_train, y_train, epochs=10)
  ```

* 케라스를 사용하면 인공신경망의 층을 직관적으로 설계할 수 있습니다

  ![image01.png](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section2/github/image01.PNG?raw=true)

<br>

### 02. Sequential 클래스의 사용 방법을 알아봅니다

* 0

  1. Sequential 객체에 층을 추가합니다

  2. add() 메서드 사용해 층을 추가합니다

     * 0

       ```python
       dense = Dense(...)
       model.add(dense)
       ```

       

     * 0

       ```python
       model = Sequential()
       model.add(Dense(...))
       model.add(Dense(...))
       ```

       

<br>

### 03. Dense 클래스의 사용 방법을 알아봅니다

* 뉴런의 개수를 지정하는 매개변수 unit

  ```python
  Dense(unit=100, ...)
  ```

  

* 활성화 함수를 지정하는 매개변수 activation

  ```python
  Dense(100, activation='sigmoid')
  ```

  

<br>

### 04. 모델의 최적화 알고리즘과 손실 함수를 설정합니다

* 최적화 알고리즘을 지정하는 매개변수 optimizer

  ```python
  model.compile(optimizer='sgd', ...)
  ```

  

* 손실 함수를 지정하는 매개변수 loss

  ```python
  model.compile(optimizer='sgd', loss='categorical_crossentropy')
  ```

  

<br>

### 05. 모델을 훈련하고 예측합니다

* 0

* 전형적인 Sequential 클래스의 사용 방법

  ```python
  model = Sequential()
  model.add(Dense(...))
  model.add(Dense(...))
  model.compile(optimizer='...', loss='...')
  model.fit(X, y, epochs=...)
  model.predict(X)
  model.evalute(X, y)
  ```

  

<br>

### 06. 케라스로 다중 분류 신경망을 만들어봅니다

1. 모델 생성하기

   ```python
   from tensorflow.keras import Sequential
   from tensorflow.keras.layers import Dense
   model = Sequential()
   ```

   

2. 은닉층과 출력층을 모델에 추가하기

   ```python
   model.add(Dense(100, activation='sigmoid', input_shape=(784,)))
   model.add(Dense(10, activation='softmax'))
   ```

   

3. 최적화 알고리즘과 손실 함수 지정하기

   ```python
   model.compile(optimizer='sgd', loss='categorical_crossentropy',
                 metrics=['accuracy'])
   ```

   

4. 모델 훈련하기

   ```python
   history = model.fit(x_train, y_train_encoded, epochs=40, validation_data=(x_val, y_val_encoded))
   ```

   

5. 손실과 정확도 그래프 그리기

   * 0

     ```python
     print(history.history.keys())	
     # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
     ```

     

   * 훈련 세트와 검증 세트의 손실 그래프

     ```python
     plt.plot(history.history['loss'])
     plt.plot(history.history['val_loss'])
     plt.ylabel('loss')
     plt.xlabel('epoch')
     plt.legend(['train_loss', 'val_loss'])
     plt.show()
     ```

     ![image02.png](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section2/github/image02.PNG?raw=true)

   * 훈련 세트와 검증 세트의 정확도

     ```python
     plt.plot(history.history['accuracy'])
     plt.plot(history.history['val_accuracy'])
     plt.ylabel('accuracy')
     plt.xlabel('epoch')
     plt.legend(['train_accuracy', 'val_accuracy'])
     plt.show()
     ```

     ![image03.png](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap07/section2/github/image03.PNG?raw=true)

6. 검증 세트 정확도 계산하기

   ```python
   loss, accuracy = model.evaluate(x_val, y_val_encoded, verbose=0)
   print(accuracy)	# 0.8629999756813049
   ```

   