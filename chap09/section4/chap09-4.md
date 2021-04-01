# chap 09-4 LSTM 순환 신경망을 만들고 텍스트를 분류합니다

2021.04.01

<br>

### 01. LSTM 셀의 구조를 알아봅니다

* LSTM 셀의 구조

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section4/image01.PNG?raw=true" style="zoom: 67%;" />

* LSTM 셀 계산 수행 과정

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section4/image02.PNG?raw=true" style="zoom: 50%;" />

* 전체 공식

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section4/image03.PNG?raw=true" style="zoom: 67%;" />

<br>

### 02. 텐서플로로 LSTM 순환 신경망 만들기

1. LSTM 순환 신경망 만들기

   ```python
   from tensorflow.keras.layers import LSTM
   
   model_lstm = Sequential()
   
   model_lstm.add(Embedding(1000, 32))
   model_lstm.add(LSTM(8))
   model_lstm.add(Dense(1, activation='sigmoid'))
   
   model_lstm.summary()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section4/image04.PNG?raw=true" style="zoom: 67%;" />

2. 모델 훈련하기

   ```python
   model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   
   history = model_lstm.fit(x_train_seq, y_train, epochs=10, batch_size=32, validation_data=(x_val_seq, y_val))
   ```

   

3. 손실 그래프와 정확도 그래프 그리기

   ```python
   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.show()
   
   plt.plot(history.history['accuracy'])
   plt.plot(history.history['val_accuracy'])
   plt.show()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section4/image05.PNG?raw=true" style="zoom: 80%;" />

4. 검증 세트 정확도 평가하기

   ```python
   loss, accuracy = model_lstm.evaluate(x_val_seq, y_val, verbose=0)
   print(accuracy)	# 0.8274
   ```

   