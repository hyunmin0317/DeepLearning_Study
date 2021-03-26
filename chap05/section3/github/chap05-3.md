# chap 05-3 규제 방법을 배우고 단일층 신경망에 적용합니다

2021.03.26

```markdown
모델이 몇 개의 데이터에 집착하면 새로운 데이터에 적응하지 못하므로 좋은 성능을 가졌다고 할 수 없는데 이를 '모델이 일반화되지 않았다'라고 말하며 이때 규제를 사용하여 가중치를 제한하면 모델이 몇 개의 데이터에 집착하지 않아 일반화 성능을 높일 수 있음

* 가중치 규제(regularization): 가중치의 값이 커지지 않도록 제한하는 기법으로 과대적합을 해결하는 대표적인 방법
```

<br>

### 01. L1 규제를 알아봅니다

![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section3/github/image01.PNG?raw=true)

* L1 규제: 손실 함수에 가중치의 절댓값인 L1 노름(norm)을 추가

* L1 규제를 경사 하강법 알고리즘에 적용하는 방법

  ```python
  w_grad += alpha * np.sign(w)
  ```

<br>

### 02. L2 규제를 알아봅니다

![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section3/github/image02.PNG?raw=true)

* L2 규제: 손실 함수에 가중치에 대한 L2 노름(norm)의 제곱을 더함

* L2 규제를 경사 하강법 알고리즘에 적용하는 방법

  ```python
  w_grad += alpha * w
  ```

* L2 규제는 가중치의 부호만 사용하는 L1 규제보다 조금 더 효과적이어서 L2 규제를 널리 사용함

<br>

### 05. L1 규제와 L2 규제 정리

* L1 규제: 그레이디언트에서 alpha에 가중치의 부호를 곱하여 그레이디언트에 더합니다.

  ```python
  w_grad += alpha * np.sign(w)
  ```

* L2 규제: 그레이디언트에서 alpha에 가중치를 곱하여 그레이디언트에 더합니다.

  ```python
  w_grad += alpha * w
  ```

<br>

### 06. 로지스틱 회귀에 규제를 적용합니다

1. 그레이디언트 업데이트 수식에 페널티 항 반영하기

   ```python
   def __init__(self, learning_rate=0.1, l1=0, l2=0):
           self.w = None
           self.b = None
           self.losses = []
           self.val_losses = []
           self.w_history = []
           self.lr = learning_rate
           self.l1 = l1
           self.l2 = l2
   ```

   * L1 규제와 L2 규제의 강도를 조절하는 매개변수 l1과 l2를 추가함 (l1과 l2의 기본값은 0)

2. fit() 메서드의 역방향 계산에 그레이디언트 페널티 항의 미분값을 더함

   ```python
   def fit(self, x, y, epochs=100, x_val=None, y_val=None):
           self.w = np.ones(x.shape[1])               # 가중치를 초기화합니다.
           self.b = 0                                 # 절편을 초기화합니다.
           self.w_history.append(self.w.copy())       # 가중치를 기록합니다.
           np.random.seed(42)                         # 랜덤 시드를 지정합니다.
           for i in range(epochs):                    # epochs만큼 반복합니다.
               loss = 0
               # 인덱스를 섞습니다
               indexes = np.random.permutation(np.arange(len(x)))
               for i in indexes:                      # 모든 샘플에 대해 반복합니다
                   z = self.forpass(x[i])             # 정방향 계산
                   a = self.activation(z)             # 활성화 함수 적용
                   err = -(y[i] - a)                  # 오차 계산
                   w_grad, b_grad = self.backprop(x[i], err) # 역방향 계산
                   # 그래디언트에서 페널티 항의 미분 값을 더합니다
                   w_grad += self.l1 * np.sign(self.w) + self.l2 * self.w
                   self.w -= self.lr * w_grad         # 가중치 업데이트
                   self.b -= b_grad                   # 절편 업데이트
                   # 가중치를 기록합니다.
                   self.w_history.append(self.w.copy())
                   # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적합니다
                   a = np.clip(a, 1e-10, 1-1e-10)
                   loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
               # 에포크마다 평균 손실을 저장합니다
               self.losses.append(loss/len(y) + self.reg_loss())
               # 검증 세트에 대한 손실을 계산합니다
               self.update_val_loss(x_val, y_val)
   ```

   * L1 규제와 L2 규제를 따로 적용하지 않고 하나의 식으로 작성하여 L1 규제와 L2규제를 동시에 수행

3. 로지스틱 손실 함수 계산에 페널티 항 추가하기

   ```python
   def reg_loss(self):
       return self.l1 * np.sum(np.abs(self.w)) + self.l2 / 2 * np.sum(self.w**2)
   ```

   * 로지스틱 손실 함수를 계산할 때 페널티 항에 대한 값을 더하기 위해 reg_loss() 메서드를 SingleLayer 클래스에 추가

   <br>

4. 검증 세트의 손실을 계산하는 update_val_loss() 메서드에서 reg_loss()를 호출하도록 수정

   ```python
   def update_val_loss(self, x_val, y_val):
           if x_val is None:
               return
           val_loss = 0
           for i in range(len(x_val)):
               z = self.forpass(x_val[i])     # 정방향 계산
               a = self.activation(z)         # 활성화 함수 적용
               a = np.clip(a, 1e-10, 1-1e-10)
               val_loss += -(y_val[i]*np.log(a)+(1-y_val[i])*np.log(1-a))
           self.val_losses.append(val_loss/len(y_val) + self.reg_loss())
   ```

5. cancer 데이터 세트에 L1 규제 적용하기

   ```python
   l1_list = [0.0001, 0.001, 0.01]
   
   for l1 in l1_list:
       lyr = SingleLayer(l1=l1)
       lyr.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val)
       
       plt.plot(lyr.losses)
       plt.plot(lyr.val_losses)
       plt.title('Learning Curve (l1={})'.format(l1))
       plt.ylabel('loss')
       plt.xlabel('epoch')
       plt.legend(['train_loss', 'val_loss'])
       plt.ylim(0, 0.3)
       plt.show()
       
       plt.plot(lyr.w, 'bo')
       plt.title('Weight (l1={})'.format(l1))
       plt.ylabel('value')
       plt.xlabel('weight')
       plt.ylim(-4, 4)
       plt.show()
   ```

   ![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section3/github/image03.PNG?raw=true)

   *  모델의 성능 확인

     ```python
     layer5 = SingleLayer(l1=0.001)
     layer5.fit(x_train_scaled, y_train, epochs=20)
     layer5.score(x_val_scaled, y_val)	# 0.978021978021978
     ```

     <br>

6. cancer 데이터 세트에 L2 규제 적용하기

   ```python
   l2_list = [0.0001, 0.001, 0.01]
   
   for l2 in l2_list:
       lyr = SingleLayer(l2=l2)
       lyr.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val)
       
       plt.plot(lyr.losses)
       plt.plot(lyr.val_losses)
       plt.title('Learning Curve (l2={})'.format(l2))
       plt.ylabel('loss')
       plt.xlabel('epoch')
       plt.legend(['train_loss', 'val_loss'])
       plt.ylim(0, 0.3)
       plt.show()
       
       plt.plot(lyr.w, 'bo')
       plt.title('Weight (l2={})'.format(l2))
       plt.ylabel('value')
       plt.xlabel('weight')
       plt.ylim(-4, 4)
       plt.show()
   ```

   ![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section3/github/image04.png?raw=true)

   *  모델의 성능 확인

     ```python
     layer6 = SingleLayer(l2=0.01)
     layer6.fit(x_train_scaled, y_train, epochs=50)
     layer6.score(x_val_scaled, y_val)	# 0.978021978021978
     ```

   * 여기서는 데이터 세트의 샘플 개수가 적어서 L1규제와 L2 규제를 적용한 모델의 성능에는 차이가 없음

   * 91개 검증 샘플 중 89개의 샘플을 올바르게 예측함

     ```python
     np.sum(layer6.predict(x_val_scaled) == y_val)	#89
     ```

7. SGDClassifier에서 규제 사용하기

   ```python
   sgd = SGDClassifier(loss='log', penalty='l2', alpha=0.001, random_state=42)
   sgd.fit(x_train_scaled, y_train)
   sgd.score(x_val_scaled, y_val)
   ```

   * 사이킷런의 SGDClassifier 클래스는 L1 규제, L2 규제를 지원하며 penalty 매개변수에 l1이나 l2 매개변수 값으로 전달하고 alpha 매개변수에 규제의 강도를 지정하여 사용함

