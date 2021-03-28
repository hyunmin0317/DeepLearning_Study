# chap 06-1 신경망 알고리즘을 벡터화하여 한 번에 전체 샘플을 사용합니다

2021.03.

```markdown
머신러닝에서는 훈련 데이터를 2차원 배열로 표현하는 경우가 많으며 이번에는 행렬 개념을 신경망 알고리즘에 도입해 보겠습니다.
```

<br>

### 01. 벡터화된 연산은 알고리즘의 성능을 올립니다

* 넘파이, 머신러닝, 딥러닝 패키지들은 다차원 배열의 계산인 행렬 연산(벡터화된 연산)을 빠르게 수행할 수 있음
* 벡터화(vectorization)된 연산을 사용하면 알고리즘의 성능을 높일 수 있음
* 배치 경사 하강법으로 성능을 올립니다
  * 확률적 경사 하강법: 기중치를 1번 업데이트할 때 1개의 샘플을 사용하므로 손실 함수의 전역 최솟값을 불안정하게 찾음
  * 배치 경사 하강법: 가중치를 1번 업데이트할 때 전체 샘플을 사용하므로 손실 함수의 전역 최솟값을 안정적으로 찾음
    * 가중치를 1번 업데이트할 때 사용되는 데이터의 개수가 많으므로 계산 비용이 많이 듦

<br>

### 02. 벡터 연산과 행렬 연산을 알아봅니다

```python
신경망에서 자주 사용하는 벡터 연산 중 하나인 점 곱(스칼라 곱)과 행렬 곱셈에 대해 알아봅니다
```

* 점 곱을 알아봅니다 (스칼라 곱)

  * 단일층 신경망에서 z를 구하는 방법 (forpass 메서드)

    ```python
    z = np.sum(x * self.w) + self.b
    ```

  * 넘파이의 원소별 곱셈 기능으로 입력과 가중치의 곱을 x * self.w으로 간단하게 표현할 수 있음

    ```python
    x = [x1, x2, ..., xn]
    w = [w1, w2, ..., wn]
    x * w = [x1*w1, x2*w2, ..., xn*wn]
    ```

  * x와 w는 벡터이며 두 벡터를 곱하여 합을 구하는 계산(np.sum(x*self.w))을 점 곱 또는 스칼라 곱이라고 함

    ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section1/github/image01.PNG?raw=true)

* 점 곱을 행렬 곱셈으로 표현합니다

  * 점 곱을 행렬 곱셈으로 표현

    ![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section1/github/image02.PNG?raw=true)

  * 점 곱을 행렬 곱셈으로 표현하여 코드 수정

    ```python
    z = np.dot(x, self.w) + self.b
    ```

* 전체 샘플에 대한 가중치 곱의 합을 행렬 곱셈으로 구합니다

  * 전체 훈련 데이터 행렬(X)를 가중치(W)와 곱하는 예

    ![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section1/github/image03.PNG?raw=true)

  * 행렬 곱셈을 통해 만들어지는 결과 행렬의 크기

    ```mathematica
    (m, n) * (n, k) = (m, k)
    ```

  * 행렬 곱셈을 넘파이의 np.dot() 함수로 구현

    ```python
    np.dot(x, w)
    ```

<br>

### 03. SingleLayer 클래스에 배치 경사 하강법 적용하기

1. 사용할 패키지 임포트 (넘파이, 맷플롯립)

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   ```

2. 위스콘신 유방암 데이터 세트를 훈련, 검증, 테스트 세트로 나누고 데이터 살펴보기

   ```python
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   
   cancer = load_breast_cancer()
   x = cancer.data
   y = cancer.target
   x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
   x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)
   ```

3. 사용할 데이터의 크기 확인

   ```python
   print(x_train.shape, x_val.shape)	# (364, 30) (91, 30)
   ```

4. 정방향 계산을 행렬 곱셈으로 표현하기

   * 정방향 계산을 행렬 곱셈으로 표현 (훈련 세트와 가중치를 곱한 다음 절편을 더함)

     ![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section1/github/image04.PNG?raw=true)

   * 벡터와 스칼라의 덧셈 연산 과정

     ![image05](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section1/github/image05.PNG?raw=true)

   <br>

5. 그레이디언트 계산 이해하기

   ![image06](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section1/github/image06.PNG?raw=true)

   * 가중치를 업데이트하기 위한 그레이디언트 계산하는 과정 (X^T=X를 전치한 배열, E=오차)
   * 샘플의 각 특성들을 오차에 곱할 수 있게 행렬을 전치하여 행과 열을 바꿈

6. forpass(), backprop() 메서드에 배치 경사 하강법 적용하기

   ```python
   def forpass(self, x):
           z = np.dot(x, self.w) + self.b        # 선형 출력을 계산합니다.
           return z
   
       def backprop(self, x, err):
           m = len(x)
           w_grad = np.dot(x.T, err) / m         # 가중치에 대한 그래디언트를 계산합니다.
           b_grad = np.sum(err) / m              # 절편에 대한 그래디언트를 계산합니다.
           return w_grad, b_grad
   ```

   * forpass() 메서드는 np.sum() 함수 대신 행렬 곱셈을 해 주는 np.dot() 함수를 사용
   * backprop() 메서드는 그레디언트의 합인 행렬 곱셈을 적용한 결과를 전체 샘플 개수로 나눠 평균 그레이디언트를 구함

7. fit() 메서드 수정하기

   ```python
   def fit(self, x, y, epochs=100, x_val=None, y_val=None):
           y = y.reshape(-1, 1)                  # 타깃을 열 벡터로 바꿉니다.
           y_val = y_val.reshape(-1, 1)
           m = len(x)                            # 샘플 개수를 저장합니다.
           self.w = np.ones((x.shape[1], 1))     # 가중치를 초기화합니다.
           self.b = 0                            # 절편을 초기화합니다.
           self.w_history.append(self.w.copy())  # 가중치를 기록합니다.
           # epochs만큼 반복합니다.
           for i in range(epochs):
               z = self.forpass(x)               # 정방향 계산을 수행합니다.
               a = self.activation(z)            # 활성화 함수를 적용합니다.
               err = -(y - a)                    # 오차를 계산합니다.
               # 오차를 역전파하여 그래디언트를 계산합니다.
               w_grad, b_grad = self.backprop(x, err)
               # 그래디언트에 페널티 항의 미분 값을 더합니다.
               w_grad += (self.l1 * np.sign(self.w) + self.l2 * self.w) / m
               # 가중치와 절편을 업데이트합니다.
               self.w -= self.lr * w_grad
               self.b -= self.lr * b_grad
               # 가중치를 기록합니다.
               self.w_history.append(self.w.copy())
               # 안전한 로그 계산을 위해 클리핑합니다.
               a = np.clip(a, 1e-10, 1-1e-10)
               # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
               loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
               self.losses.append((loss + self.reg_loss()) / m)
               # 검증 세트에 대한 손실을 계산합니다.
               self.update_val_loss(x_val, y_val)
   ```

   * 배치 경사 하강법에서는 forpass()와 backprop() 메서드에서 전체 샘플을 한번에 계산하기 때문에 두 번째 for문이 삭제됨
   * 평균 손실을 구하기 위해 np.sum() 함수로 각 샘플의 손실을 더한 후 전체 샘플의 개수로 나눔

   <br>

8. 나머지 메서드 수정하기

   * predict() 메서드와 update_val_loss() 메서드 수정

     ```python
     def predict(self, x):
             z = self.forpass(x)      # 정방향 계산을 수행합니다.
             return z > 0             # 스텝 함수를 적용합니다.
         
     def update_val_loss(self, x_val, y_val):
             z = self.forpass(x_val)            # 정방향 계산을 수행합니다.
             a = self.activation(z)             # 활성화 함수를 적용합니다.
             a = np.clip(a, 1e-10, 1-1e-10)     # 출력 값을 클리핑합니다.
             # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
             val_loss = np.sum(-(y_val*np.log(a) + (1-y_val)*np.log(1-a)))
             self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))
     ```

     * update_val_loss() 메서드에서도 검증 손실 val_loss를 계산할 때 np.sum() 함수를 적용

   * 배치 경사 하강법을 적용한 SingleLayer 클래스의 전체 코드

     ```python
     class SingleLayer:
         
         def __init__(self, learning_rate=0.1, l1=0, l2=0):
             self.w = None              # 가중치
             self.b = None              # 절편
             self.losses = []           # 훈련 손실
             self.val_losses = []       # 검증 손실
             self.w_history = []        # 가중치 기록
             self.lr = learning_rate    # 학습률
             self.l1 = l1               # L1 손실 하이퍼파라미터
             self.l2 = l2               # L2 손실 하이퍼파라미터
     
         def forpass(self, x):
             z = np.dot(x, self.w) + self.b        # 선형 출력을 계산합니다.
             return z
     
         def backprop(self, x, err):
             m = len(x)
             w_grad = np.dot(x.T, err) / m         # 가중치에 대한 그래디언트를 계산합니다.
             b_grad = np.sum(err) / m              # 절편에 대한 그래디언트를 계산합니다.
             return w_grad, b_grad
     
         def activation(self, z):
             z = np.clip(z, -100, None)            # 안전한 np.exp() 계산을 위해
             a = 1 / (1 + np.exp(-z))              # 시그모이드 계산
             return a
             
         def fit(self, x, y, epochs=100, x_val=None, y_val=None):
             y = y.reshape(-1, 1)                  # 타깃을 열 벡터로 바꿉니다.
             y_val = y_val.reshape(-1, 1)
             m = len(x)                            # 샘플 개수를 저장합니다.
             self.w = np.ones((x.shape[1], 1))     # 가중치를 초기화합니다.
             self.b = 0                            # 절편을 초기화합니다.
             self.w_history.append(self.w.copy())  # 가중치를 기록합니다.
             # epochs만큼 반복합니다.
             for i in range(epochs):
                 z = self.forpass(x)               # 정방향 계산을 수행합니다.
                 a = self.activation(z)            # 활성화 함수를 적용합니다.
                 err = -(y - a)                    # 오차를 계산합니다.
                 # 오차를 역전파하여 그래디언트를 계산합니다.
                 w_grad, b_grad = self.backprop(x, err)
                 # 그래디언트에 페널티 항의 미분 값을 더합니다.
                 w_grad += (self.l1 * np.sign(self.w) + self.l2 * self.w) / m
                 # 가중치와 절편을 업데이트합니다.
                 self.w -= self.lr * w_grad
                 self.b -= self.lr * b_grad
                 # 가중치를 기록합니다.
                 self.w_history.append(self.w.copy())
                 # 안전한 로그 계산을 위해 클리핑합니다.
                 a = np.clip(a, 1e-10, 1-1e-10)
                 # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
                 loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
                 self.losses.append((loss + self.reg_loss()) / m)
                 # 검증 세트에 대한 손실을 계산합니다.
                 self.update_val_loss(x_val, y_val)
         
         def predict(self, x):
             z = self.forpass(x)      # 정방향 계산을 수행합니다.
             return z > 0             # 스텝 함수를 적용합니다.
         
         def score(self, x, y):
             # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환합니다.
             return np.mean(self.predict(x) == y.reshape(-1, 1))
         
         def reg_loss(self):
             # 가중치에 규제를 적용합니다.
             return self.l1 * np.sum(np.abs(self.w)) + self.l2 / 2 * np.sum(self.w**2)
         
         def update_val_loss(self, x_val, y_val):
             z = self.forpass(x_val)            # 정방향 계산을 수행합니다.
             a = self.activation(z)             # 활성화 함수를 적용합니다.
             a = np.clip(a, 1e-10, 1-1e-10)     # 출력 값을 클리핑합니다.
             # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
             val_loss = np.sum(-(y_val*np.log(a) + (1-y_val)*np.log(1-a)))
             self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))
     ```

   <br>

9. 훈련 데이터 표준화 전처리하기

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   scaler.fit(x_train)
   x_train_scaled = scaler.transform(x_train)
   x_val_scaled = scaler.transform(x_val)
   ```

   * 안정적인 학습을 위해 사이킷런의 StandardScaler 클래스를 사용해 데이터 세트의 특성을 평균이 0, 표준편차가 1이 되도록 변환
   * 변환기(transformer): 데이터 전처리에 관련된 클래스들로 sklearn.preprocessing 모듈 아래에 있음
   * StarndardScaler 클래스로 scaler 객체를 만든 다음 fit() 메서드를 통해 변환 규칙을 익히고 transform() 메서드로 데이터를 표준화 전처리한 후 훈련 세트와 검증 세트에 표준화를 적용하여 x_train_scaled, x_val_scaled를 준비

10. 배치 경사 하강법 적용

    ```python
    single_layer = SingleLayer(l2=0.01)
    single_layer.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val, epochs=10000)
    single_layer.score(x_val_scaled, y_val)	# 0.978021978021978
    ```

    * L2 규제 매개변수의 값을 0.01로 지정하고 에포크 매개변수의 기본값을 100에서 10000으로 늘림
    * 확률적 경사 하강법과 배치 경사 하강법은 에포크마다 가중치 업데이트를 하는 횟수에 차이가 있으므로 에포크를 늘림

11. 검증 세트로 성능 측정하고 그래프로 비교하기

    ```python
    plt.ylim(0, 0.3)
    plt.plot(single_layer.losses)
    plt.plot(single_layer.val_losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.show()
    ```

    ![image07](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section1/github/image07.PNG?raw=true)

    * 훈련 손실과 검증 손실을 그래프로 출력하여 확률적 경사 하강법과 비교해보면 손실값이 안정적으로 감소하는걸 확인할 수 있음

12. 가중치의 변화를 그래프로 나타내어 결과의 원인 분석

    ```python
    w2 = []
    w3 = []
    for w in single_layer.w_history:
        w2.append(w[2])
        w3.append(w[3])
    plt.plot(w2, w3)
    plt.plot(w2[-1], w3[-1], 'ro')
    plt.xlabel('w[2]')
    plt.ylabel('w[3]')
    plt.show()
    ```

    ![image08](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section1/github/image08.PNG?raw=true)

    * 배치 경사하강법을 적용하면 가중치를 찾는 경로가 곡선의 형태이며 가중치의 변화가 연속적이므로 손실값도 안정적으로 수렴됨