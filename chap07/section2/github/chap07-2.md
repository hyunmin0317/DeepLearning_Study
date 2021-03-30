# chap 07-2 텐서플로와 케라스를 사용하여 신경망을 만듭니다

2021.03.30

`높은 성능을 가진 인공신경망 모델을 만들기 위해서는 전문적인 라이브러리를 사용해야 하며 대표적인 딥러닝 패키지인 구글의 텐서플로를 사용하고 이를 쉽게 사용하기 위한 케라스 API(Keras API)를 통해 인공신경망을 만들어 보겠습니다.`

<br>

### 01. 케라스에 대해 알아봅니다

* 케라스: 딥러닝 패키지를 편리하게 사용하기 위해 만들어진 래퍼(Wrapper) 패키지

* 간단한 신경망을 텐서플로와 케라스로 구현한 예시

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

* 위의 신경망을 케라스로 구현

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

  * 케라스는 인공신경망 모델을 만들기 위한 Sequential 클래스와 완전 연결층을 만들기 위한 Dense 클래스를 제공
    * Sequential 클래스: 순차적으로 층을 쌓은 신경망 모델
    * Dense 클래스: 모델에 포함된 완전 연결층

<br>

### 02. Sequential 클래스의 사용 방법을 알아봅니다

* 완전 연결 신경망을 만들려면 Sequential 클래스와 Dense 클래스를 함께 사용함

* Sequential 클래스를 사용하는 방법 (2가지)

  1. Sequential 객체에 층을 추가합니다

     ```python
     from tensorflow.keras import Sequential
     from tensorflow.keras.layers import Done
     model = Sequential([Dense(...), ...])
     ```

     * Sequential 클래스로 객체를 생성할 때 Dense 클래스로 만든 층을 추가할 수 있음

  2. add() 메서드 사용해 층을 추가합니다

     * Sequential 클래스의 add() 메서드를 사용하여 층 추가

       ```python
       dense = Dense(...)
       model.add(dense)
       ```

     * Dense 클래스의 객체를 만들자마자 Dense 클래스의 객체를 변수에 할당하여 전달

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

  * Dense 클래스에 전달해야 하는 첫 번째 매개변수로 층의 유닛(unit) 개수

* 활성화 함수를 지정하는 매개변수 activation

  ```python
  Dense(100, activation='sigmoid')
  ```

  * Dense 클래스의 두 번째 매개변수로 활성화 함수를 지정
  * 기본값은 None이며 activation에는 sigmoid, softmax, tanh, relu 등 많은 함수를 적용할 수 있음 

<br>

### 04. 모델의 최적화 알고리즘과 손실 함수를 설정합니다

* 모델을 훈련하기 위해서는 최적화 알고리즘이나 손실 함수를 지정해야 함

* 다중 분류의 최적화 알고리즘은 경사 하강법 알고리즘을 사용하고 손실 함수는 크로스 엔트로피 손실 함수를 사용 

* 케라스에서 최적화 알고리즘과 손실 함수를 지정하는 방법
  * 최적화 알고리즘을 지정하는 매개변수 optimizer

    ```python
    model.compile(optimizer='sgd', ...)	
    ```
    * Sequential 클래스의 compile() 메서드를 사용하여 최적화 알고리즘과 손실 함수를 지정
    * 최적화 알고리즘은 매개변수 optimizer를 사용하고 기본 경사 하강법은 'sgd'로 지정하고 학습률의 기본값은 0.01

  * 손실 함수를 지정하는 매개변수 loss

    ```python
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    ```

    * loss 매개변수를 제곱 오차의 경우 mse, 로지스틱 손실 함수의 경우 binary_crossentropy, 다중 분류 신경망은 categorical_crossentropy로 지정

<br>

### 05. 모델을 훈련하고 예측합니다

* Sequential 클래스의 fit() 메서드와 predict() 메서드로 모델은 훈련하고 예측하며 evaluate() 메서드로 평가

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

   * tensorflow.keras 모듈 안에 있는 Sequential 클래스와 Dense 클래스를 임포트하고 Sequential 객체(모델)를 생성

2. 은닉층과 출력층을 모델에 추가하기

   ```python
   model.add(Dense(100, activation='sigmoid', input_shape=(784,)))
   model.add(Dense(10, activation='softmax'))
   ```

   * 은닉층의 유닛 개수는 100개, 출력층의 유닛 개수는 10개인 신경망을 만들기 위해 첫 번째 매개변수를 100, 10으로 지정
   * 활성화 함수는 은닉층은 시그모이드 함수, 출력층은 소프트맥스 함수이므로 두 번째 매개변수를 'sigmoid', 'softmax'로 지정

3. 최적화 알고리즘과 손실 함수 지정하기

   ```python
   model.compile(optimizer='sgd', loss='categorical_crossentropy',
                 metrics=['accuracy'])
   ```

   * 최적화 알고리즘으로는 경사 하강법을, 손실 함수는 크로스 엔트로피 손실 함수를 사용하므로 optimizer와 loss에 각각 'sgd', 'categorical_crossentropy'로 지정
   * 훈련 과정 기록으로 정확도를 남기기 위해 metrics 매개변수를 추가하고 정확도에 대한 기록이 필요하므로 'accuracy'를 추가

4. 모델 훈련하기

   ```python
   history = model.fit(x_train, y_train_encoded, epochs=40, validation_data=(x_val, y_val_encoded))
   ```

   * fit() 메서드를 통해 모델을 40번의 에포크 동안 훈련하며 검증 세트에 대한 손실과 정확도를 계산

5. 손실과 정확도 그래프 그리기

   * history 딕셔너리의 측정 지표

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

   * 손실 그래프는 일정한 수준으로 감소하는 추세를 보이고 정확도 그래프도 점진적으로 증가하고 있음

6. 검증 세트 정확도 계산하기

   ```python
   loss, accuracy = model.evaluate(x_val, y_val_encoded, verbose=0)
   print(accuracy)	# 0.8629999756813049
   ```

   * evaluate() 메서드를 사용하여 손실값과 metrics 매개변수에 추가한 측정 지표를 계산하여 반환