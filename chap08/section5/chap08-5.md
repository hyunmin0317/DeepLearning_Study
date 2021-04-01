# chap 08-5 케라스로 합성곱 신경망을 만듭니다

2021.04.02

`케라스를 사용하여 합성곱과 풀링의 스트라이드, 패딩 등의 개념을 적용한 합성곱 신경망을 구현해 보겠습니다.`

<br>

### 01. 케라스로 합성곱 신경망 만들기

`케라스의 합성곱층은 Conv2D 클래스, 최대 풀링은 MaxPooling2D 클래스, 특성 맵을 일렬로 펼칠 땐느 Flatten 클래스를 사용`

1. 필요한 클래스들을 임포트하기

   ```python
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
   ```

2. 합성곱층 쌓기

   ```python
   conv1 = tf.keras.Sequential()
   conv1.add(Conv2D(10, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
   ```

3. 풀링층 쌓기

   ```python
   conv1.add(MaxPooling2D((2, 2)))
   ```

4. 완전 연결층에 주입할 수 있도록 특성 맵 펼치기

   ```python
   conv1.add(Flatten())
   ```

5. 완전 연결층 쌓기

   ```python
   conv1.add(Dense(100, activation='relu'))
   conv1.add(Dense(10, activation='softmax'))
   ```

6. 모델 구조 살펴보기

   ```python
   conv1.summary()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section5/image01.PNG?raw=true" alt="image01.PNG" style="zoom: 80%;" />

<br>

### 02. 합성곱 신경망 모델 훈련하기

1. 모델을 컴파일한 다음 훈련

   ```python
   conv1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   ```

2. 아담 옵티마이저 사용하기

   ```python
   history = conv1.fit(x_train, y_train_encoded, epochs=20, validation_data=(x_val, y_val_encoded))
   ```

3. 손실 그래프와 정확도 그래프 확인하기

   ```python
   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train_loss', 'val_loss'])
   plt.show()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section5/image02.PNG?raw=true" alt="image02.PNG" style="zoom: 80%;" />

<br>

### 03. 드롭아웃을 알아봅니다

* 축구 선수를 출전 목록에서 무작위로 제외하는 것이 드롯아웃입니다

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section5/image03.PNG?raw=true" alt="image03.PNG" style="zoom: 80%;" />

* 텐서플로에서는 드롭아웃의 비율만큼 뉴런의 출력을 높입니다

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section5/image04.PNG?raw=true" alt="image04.PNG" style="zoom: 80%;" />

<br>

### 04. 드롭아웃 적용해 합성곱 신경망을 구현합니다

`텐서플로에서 드롭아웃을 적용하려면 간단히 Dropout 클래스를 추가하며 매개변수에 드롭아웃될 비율을 실수로 지정함`

1. 케라스로 만든 합성곱 신경망에 드롭아웃 적용하기

   ```python
   from tensorflow.keras.layers import Dropout
   
   conv2 = tf.keras.Sequential()
   conv2.add(Conv2D(10, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
   conv2.add(MaxPooling2D((2, 2)))
   conv2.add(Flatten())
   conv2.add(Dropout(0.5))
   conv2.add(Dense(100, activation='relu'))
   conv2.add(Dense(10, activation='softmax'))
   ```

2. 드롭아웃층 확인하기

   ```
   conv2.summary()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section5/image05.PNG?raw=true" alt="image05.PNG" style="zoom: 80%;" />

   <br>

3. 훈련하기

   ```python
   conv2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   history = conv2.fit(x_train, y_train_encoded, epochs=20, validation_data=(x_val, y_val_encoded))
   ```

4. 손실 그래프와 정확도 그래프 그리기

   * 손실 그래프

     ```python
     plt.plot(history.history['loss'])
     plt.plot(history.history['val_loss'])
     plt.ylabel('loss')
     plt.xlabel('epoch')
     plt.legend(['train_loss', 'val_loss'])
     plt.show()
     ```

   * 정확도 그래프

     ```python
     plt.plot(history.history['accuracy'])
     plt.plot(history.history['val_accuracy'])
     plt.ylabel('accuracy')
     plt.xlabel('epoch')
     plt.legend(['train_accuracy', 'val_accuracy'])
     plt.show()
     ```

   * 그래프 결과

     <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap08/section5/image06.PNG?raw=true" alt="image06.PNG" style="zoom: 67%;" />

