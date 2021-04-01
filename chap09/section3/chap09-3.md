# chap 09-3 텐서플로로 순환 신경망을 만듭니다

<br>

### 01. SimpleRNN 클래스로 순환 신경망 만들기

1. 순환 신경망에 필요한 클래스 임포트하기

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, SimpleRNN
   ```

2. 모델 만들기

   ```python
   model = Sequential()
   
   model.add(SimpleRNN(32, input_shape=(100, 100)))
   model.add(Dense(1, activation='sigmoid'))
   
   model.summary()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section3/image01.PNG?raw=true" alt="image01.PNG" style="zoom:80%;" />

3. 모델 컴파일하고 훈련시키기

   ```python
   model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
   
   history = model.fit(x_train_onehot, y_train, epochs=20, batch_size=32, validation_data=(x_val_onehot, y_val))
   ```

   <br>

4. 훈련, 검증 세트에 대한 손실 그래프와 정확도 그래프 그리기

   ```python
   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.show()
   
   plt.plot(history.history['accuracy'])
   plt.plot(history.history['val_accuracy'])
   plt.show()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section3/image02.PNG?raw=true" alt="image02.PNG" style="zoom:80%;" />

5. 검증 세트 정확도 평가하기

   ```python
   loss, accuracy = model.evaluate(x_val_onehot, y_val, verbose=0)
   print(accuracy)	# 0.6899999976158142
   ```

<br>

### 02. 임베딩층으로 순환 신경망 모델 성능 높이기

<img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section3/image03.PNG?raw=true" alt="image03.PNG" style="zoom: 50%;" />

1. Embedding 클래스 임포트하기

   ```python
   from tensorflow.keras.layers import Embedding
   ```

2. 훈련 데이터 준비하기

   ```python
   (x_train_all, y_train_all), (x_test, y_test) = imdb.load_data(skip_top=20, num_words=1000)
   
   for i in range(len(x_train_all)):
       x_train_all[i] = [w for w in x_train_all[i] if w > 2]
       
   x_train = x_train_all[random_index[:20000]]
   y_train = y_train_all[random_index[:20000]]
   x_val = x_train_all[random_index[20000:]]
   y_val = y_train_all[random_index[20000:]]
   ```

3. 샘플 길이 맞추기

   ```python
   maxlen=100
   x_train_seq = sequence.pad_sequences(x_train, maxlen=maxlen)
   x_val_seq = sequence.pad_sequences(x_val, maxlen=maxlen)
   ```

   <br>

4. 모델 만들기

   ```python
   model_ebd = Sequential()
   
   model_ebd.add(Embedding(1000, 32))
   model_ebd.add(SimpleRNN(8))
   model_ebd.add(Dense(1, activation='sigmoid'))
   
   model_ebd.summary()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section3/image04.PNG?raw=true" alt="image04.PNG" style="zoom:80%;" />

5. 모델 컴파일하고 훈련시키기

   ```python
   model_ebd.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   
   history = model_ebd.fit(x_train_seq, y_train, epochs=10, batch_size=32, validation_data=(x_val_seq, y_val))
   ```

   

6. 손실 그래프와 정확도 그래프 그리기

   ```python
   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.show()
   
   plt.plot(history.history['accuracy'])
   plt.plot(history.history['val_accuracy'])
   plt.show()
   ```

   <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section3/image05.PNG?raw=true" alt="image06.PNG" style="zoom:80%;" />

7. 검증 세트 정확도 평가하기

   ```python
   loss, accuracy = model_ebd.evaluate(x_val_seq, y_val, verbose=0)
   print(accuracy)	# 0.8055999875068665
   ```

   