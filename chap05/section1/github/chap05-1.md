# chap 05-1 검증 세트를 나누고 전처리 과정을 배웁니다

2021.03.26

<br>

### 01. 테스트 세트로 모델을 튜닝합니다

* 로지스틱 회귀로 모델 훈련하고 평가하기

  * 데이터 세트를 읽어 들여 훈련 세트와 테스트 세트로 나누기

    ```python
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target
    x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
    ```

  * SGDClassifier 클래스를 이용하여 로지스틱 회귀 모델을 훈련

    ```python
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(loss='log', random_state=42)
    sgd.fit(x_train_all, y_train_all)
    sgd.score(x_test, y_test)	# 0.8333333333333334
    ```

    * fit() 메서드로 모델을 훈련하고 score() 메서드로 성능을 평가
    * 하이퍼파라미터(hyperparameter): 사용자가 직접 선택해야 하는 값으로 알아서 학습되지 않는 loss와 같은 매개변수를 말함

* 서포트 벡터 머신으로 모델 훈련하고 평가하기

  * SGDClassifier 클래스의 loss 매개변수를 log에서 hinge로 바꿈

    ```python
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(loss='hinge', random_state=42)
    sgd.fit(x_train_all, y_train_all)
    sgd.score(x_test, y_test)	# 0.9385964912280702
    ```

    * 선형 서포트 벡터 머신(SVM) 문제를 푸는 모델이 만들어짐
    * SVM(Support Vector Machine): 훈련 데이터의 클래스를 구분하는 경계선을 찾는 작업

* SGDClassifier 클래스의 매개변수들을 바꾸며 '모델을 튜닝'하면 테스트 성능을 높일 수 있지만 실전에서는 좋은 성능을 내지 못함  

<br>

### 02. 테스트 세트로 모델을 튜닝하면 실전에서 좋은 성능을 기대하기 어렵습니다

* 테스트 세트로 모델을 튜닝하면 '테스트 세트에 대해서만 좋은 성능을 보여주는 모델'이 만들어지고 실전에서 같은 성능을 기대하기 어려우며 이 현상을 '테스트 세트의 정보가 모델에 새어 나갔다' 라고 함
* 테스트 세트로 모델을 튜닝하면 모델의 일반화 성능(generalization performance)이 왜곡됨

<br>

### 03. 검증 세트를 준비합니다 

* 전체 데이터 세트를 훈련, 검증 , 테스트 세트로 나누고 모델을 훈련해야 함

  * 검증 세트(validation set): 모델을 튜닝하는 용도의 세트로 훈련 세트를 조금 떼어 만듦

* SGDClassifier 클래스로 만든 모델을 훈련

  1. 데이터 세트 준비하기 (위스콘신 유방암 데이터 사용)

     ```python
     from sklearn.datasets import load_breast_cancer
     from sklearn.model_selection import train_test_split
     cancer = load_breast_cancer()
     x = cancer.data
     y = cancer.target
     x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
     ```

  2. 검증 세트 분할하기

     ```python
     x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)
     print(len(x_train), len(x_val))	# 364 91
     ```

     * 전체 데이터 세트를 8:2로 나누어 훈련 세트와 테스트 세트를 만들고 훈련 세트를 8:2로 나누어 훈련 세트와 검증 세트를 만듦

  3. 검증 세트 사용해 모델 평가하기

     ```python
     sgd = SGDClassifier(loss='log', random_state=42)
     sgd.fit(x_train, y_train)
     sgd.score(x_val, y_val)	# 0.6923076923076923
     ```

     * 훈련 세트의 크기가 줄어들어 성능이 감소했고 위의 데이터 세트는 샘플 개수가 적은 편으로 검증 세트의 비율이나 매개변수의 값을 조금만 조절해도 성능 평가 점수가 크게 변함
     * 데이터 양이 적은 경우 검증 세트를 나누지 않는 교차 검증이라는 방법을 사용하지만 대부분 훈련, 검증, 테스트 세트로 나눔

<br>

### 04. 데이터 전처리와 특성의 스케일을 알아봅니다

```python
데이터 전처리(data preprocessing): 데이터를 적절히 가공하는 과정
    * 실전에서 수집된 데이터의 형태가 균일하지 않거나 누락된 값이 있을 때 데이터를 적절히 가공하는 데이터 전처리 과정이 필요함
```

* 특성의 스케일은 알고리즘에 영향을 줍니다

  * 이 책에서는 제대로 가공되지 않은 데이터를 다루지 않지만 특성의 스케일(scale)이 다른 경우 또한 데이터 전처리 과정이 필요함

  * 특성의 스케일(값의 범위)이 다른 경우

    |        | 당도 | 무게 |
    | :----: | :--: | :--: |
    | 사과 1 |  4   | 540  |
    | 사과 2 |  8   | 700  |
    | 사과 3 |  2   | 480  |

    * 사과의 당도 범위는 1-10이고 무게의 범위는 500-1000으로 두 특성의 스케일 차이가 큼
    * 어떤 알고리즘은(신경망 알고리즘 등) 스케일에 민감하여 모델의 성능에 영향을 줌

<br>

### 05. 스케일을 조정하지 않고 모델을 훈련해 볼까요?

1. 훈련 데이터 준비하고 스케일 비교하기

   ```python
   print(cancer.feature_names[[2,3]])
   plt.boxplot(x_train[:, 2:4])
   plt.xlabel('feature')
   plt.ylabel('value')
   plt.show()	# ['mean perimeter' 'mean area']
   ```

   ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section1/github/image01.PNG?raw=true)

   * 위스콘신 유방암 데이터의 mean perimeter는 주로 100-200 사이에 값들이 위치한 반면 mean area는 200-2000 사이에 값들이 집중되어 있음

2. 가중치를 기록할 변수와 학습률 파라미터 추가하기

   ```python
   def __init__(self, learning_rate=0.1, l1=0, l2=0):
           self.w = None
           self.b = None
           self.losses = []
           self.val_losses = []
           self.w_history = []
           self.lr = learning_rate
   ```

   * 추가한 파라미터 설명
     * learning_rate: 학습률 파라미터로 가중치의 업데이트 양을 조절
     * w_history: 인스턴스 변수로 에포크마다 가중치의 값을 저장

   <br>

3. 가중치 기록하고 업데이트 양 조절하기

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

   * 가중치가 바뀔 때마다 w_history 리스트에 가중치를 기록 (가중치가 바뀔 때마다 그 값을 복사하여 w_history 리스트에 추가)
   * w_grad에 학습률 self.lr을 곱하여 가중치 업데이트 양 조절

4. 모델 훈련하고 평가하기

   ```python
   layer1 = SingleLayer()
   layer1.fit(x_train, y_train)
   layer1.score(x_val, y_val)	# 0.9120879120879121
   ```

5. 가중치의 값을 그래프로 나타내기

   ```python
   w2 = []
   w3 = []
   for w in layer1.w_history:
       w2.append(w[2])
       w3.append(w[3])
   plt.plot(w2, w3)
   plt.plot(w2[-1], w3[-1], 'ro')
   plt.xlabel('w[2]')
   plt.ylabel('w[3]')
   plt.show()
   ```

   ![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section1/github/image02.PNG?raw=true)

   * 100번의 에포크 동안 변경된 가중치가 모두 인스턴스 변수 w_history에 기록되어 있고 세 번째, 네 번째 요소는 각각 mean perimeter와 mean area 특성에 대한 가중치이며 이를 그래프로 그리면 다음과 같음 (최종으로 결정된 가중치는 점으로 표시)
   * 가중치의 최적값에 도달하는 동안 w3 값이 요동치므로 모델이 불안정하게 수렴하며 이를 줄이기 위해 스케일을 조정해야 함

<br>

### 06. 스케일을 조정해 모델을 훈련합니다

```markdown
스케일을 조정하는 방법은 많지만 신경망에서는 스케일 조정 방법 중 표준화(standardization)를 많이 사용함
* 표준화는 특성값에서 평균을 빼고 표준 편차로 나누면 됨
```

1. 넘파이로 표준화 구현하기

   ```python
   train_mean = np.mean(x_train, axis=0)
   train_std = np.std(x_train, axis=0)
   x_train_scaled = (x_train - train_mean) / train_std
   ```

   * 넘파이의 mean(), std() 함수를 사용하여 평균과 표준 편차를 계산하여 표준화를 쉽게 구현

2. 모델 훈련하기

   ```python
   layer2 = SingleLayer()
   layer2.fit(x_train_scaled, y_train)
   w2 = []
   w3 = []
   for w in layer2.w_history:
       w2.append(w[2])
       w3.append(w[3])
   plt.plot(w2, w3)
   plt.plot(w2[-1], w3[-1], 'ro')
   plt.xlabel('w[2]')
   plt.ylabel('w[3]')
   plt.show()
   ```

   ![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section1/github/image03.PNG?raw=true)

   * 스케일을 조정한 데이터 세트로 단일층 신경망을 다시 훈련시키고 가중치를 그래프로 그림
   * w2와 w3의 변화 비율이 비슷하기 때문에 대각선 방향으로 가중치가 이동하며 최적값에 빠르게 근접하고 있음을 확인할 수 있음

3. 모델 성능 평가하기

   ```python
   layer2.score(x_val, y_val)	# 0.37362637362637363
   ```

4. 검증 세트도 표준화 전처리를 적용

   ```python
   val_mean = np.mean(x_val, axis=0)
   val_std = np.std(x_val, axis=0)
   x_val_scaled = (x_val - val_mean) / val_std
   layer2.score(x_val_scaled, y_val)	# 0.967032967032967
   ```

   * 검증 세트의 스케일을 바꾸지 않아 성능이 좋지 않으므로 검증 세트도 표준화 전처리를 적용함

<br>

### 07. 스케일을 조정한 다음에 실수하기 쉬운 함정을 알아봅니다

````markdown
훈련 세트와 검증 세트가 다른 비율로 스케일이 조정된 경우를 해결하는 방법
````

1. 원본 훈련 세트와 검증 세트로 산점도 그리기

   ```python
   plt.plot(x_train[:50, 0], x_train[:50, 1], 'bo')
   plt.plot(x_val[:50, 0], x_val[:50, 1], 'ro')
   plt.xlabel('feature 1')
   plt.ylabel('feature 2')
   plt.legend(['train set', 'val. set'])
   plt.show()
   ```

   ![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section1/github/image04.PNG?raw=true)

   * 파란 점이 훈련 세트이고 빨간 점이 검증 세트

2. 전처리한 훈련 세트와 검증 세트로 산점도 그리기

   ```python
   plt.plot(x_train_scaled[:50, 0], x_train_scaled[:50, 1], 'bo')
   plt.plot(x_val_scaled[:50, 0], x_val_scaled[:50, 1], 'ro')
   plt.xlabel('feature 1')
   plt.ylabel('feature 2')
   plt.legend(['train set', 'val. set'])
   plt.show()
   ```

   ![image05](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section1/github/image05.PNG?raw=true)

   * 훈련 세트와 검증 세트가 각각 다른 비율로 변환되어 점과 점 사이의 거리가 변환된 이후에 그대로 유지되지 않음
   * 훈련 세트와 검증 세트를 각각 다른 비율로 전처리했기 때문에 점과 점 사이의 거리가 달라짐

3. 올바르게 검증 세트 전처리하기

   ```python
   x_val_scaled = (x_val - train_mean) / train_std
   plt.plot(x_train_scaled[:50, 0], x_train_scaled[:50, 1], 'bo')
   plt.plot(x_val_scaled[:50, 0], x_val_scaled[:50, 1], 'ro')
   plt.xlabel('feature 1')
   plt.ylabel('feature 2')
   plt.legend(['train set', 'val. set'])
   plt.show()
   ```

   ![image06](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section1/github/image06.PNG?raw=true)

   * 검증 세트와 훈련 세트의 스케일과 다른 비율로 조정되면 모델에 적용된 알고리즘들이 검증 세트의 샘플 데이터를 잘못 인식
   * 훈련 세트의 평균, 표준 편차를 사용하여 검증 세트를 변환하여 검증 세트를 훈련 세트와 같은 비율로 전처리해야 함
   * 원본 데이터의 산점도와 스케일 조정 이후의 산점도가 같아짐 (검증 세트와 훈련 세트가 동일한 비율로 변환됨)

4. 모델 평가하기

   ```python
   layer2.score(x_val_scaled, y_val)	# 0.967032967032967
   ```