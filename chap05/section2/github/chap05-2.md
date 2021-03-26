# chap 05-2 과대적합과 과소적합을 알아봅니다

2021.03.26

<br>

### 01. 학습 곡선을 통해 과대적합과 과소적합을 알아봅니다

* 훈련 세트의 크기와 과대적합, 과소적합 분석하기

  ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section2/github/image01.PNG?raw=true)

  * 과대적합: 모델이 훈련 세트에서는 좋은 성능을 내지만 검증 세트에서는 낮은 성능을 내는 경우
    * 첫 번째 학습 곡선의 경우로 훈련 세트와 검증 세트에서 측정한 성능의 간격이 크며 '분산이 크다(high variance)' 라고도 말함
    * 훈련 세트에 다양한 패턴의 샘플이 없는 경우 검증 세트에 제대로 적응하지 못하여 발생
    * 더 많은 훈련 샘플을 모으거나 모델이 훈련에 집착하지 않도록 가중치를 제한하여 '모델의 복잡도'를 낮추면 성능이 향상됨
  * 과소적합: 훈련 세트와 검증 세트의 성능에는 차이가 크지 않지만 모두 낮은 성능을 내는 경우
    * 두 번째 학습 곡선의 경우로 훈련 세트와 검증 세트에서 측정한 성능의 간격은 가깝지만 성능 자체가 낮음
    * 모델이 충분히 복잡하지 않아 훈련 데이터에 있는 패턴을 모두 잡아내지 못하여 발생
    * 복잡도가 더 높은 모델을 사용하거나 가중치의 규제를 완화하면 성능이 향상됨
  * 세 번째 학습 곡선은 과대적합과 과소적합 사이에서 절충점을 찾은 모습

<br>

* 에포크와 손실 함수의 그래프로 과대적합과 과소적합 분석하기

  ![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section2/github/image02.PNG?raw=true)

  * 왼쪽 그래프는 검증 세트의 손실과 훈련 세트의 손실을 나타낸 것으로 에포크가 진행될수록 훈련 세트의 손실은 감소하지만 검증 세트의 손실은 최적점을 지나면 모델이 과대적합되어 오히려 상승함
  * 최적점 이전에는 훈련 세트와 검증 세트의 손실이 비슷한 간격을 유지하는데 이때 학습을 중지하면 과소적합된 모델이 만들어짐
  * 오른쪽 그래프는 정확도에 대한 그래프인데 손실에 대한 그래프와 의미가 동일함 

* 모델 복잡도와 손실 함수의 그래프로 과대적합과 과소적합 분석하기

  ![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section2/github/image03.PNG?raw=true)
  * 모델 복잡도: 모델이 가진 학습 가능한 가중치 개수를 말하며 층이나 유닛의 개수가 많아지면 복잡도가 높은 모델이 만들어짐

* 좋은 성능을 내는 모델을 만들기 위해서는 여러 조건이 필요하며 적절한 에포크 횟수를 찾아보겠습니다.

<br>

### 02. 적절한 편향-분산 트레이드오프를 선택합니다

* 편향-분산 트레이드오프(bias-variance tradeoff): 과소적합된 모델(편향)과 과대적합된 모델(분산) 사이의 관계

  * 하나를 얻기 위해서는 다른 하나를 희생해야 함(트레이드오프)
  * 분산이나 편향이 너무 커지지 않도록 적절한 중간 지점을 선택해야 함 (적절한 편향-분산 트레이드오프를 선택)

* 경사 하강법의 에포크 횟수에 대한 모델의 손실을 그래프로 그려 '적절한 편향-분산 트레이드오프' 선택하기

  1. 검증 손실을 기록하기 위한 변수 추가하기

     ```python
     def __init__(self, learning_rate=0.1, l1=0, l2=0):
             self.w = None
             self.b = None
             self.losses = []
             self.val_losses = []
             self.w_history = []
             self.lr = learning_rate
     ```

     * 검증 세트에 대한 손실을 저장하기 위해 self.val_losses 인스턴스 변수를 추가

  2. fit() 메서드에 검증 세트를 전달받을 수 있도록 매개변수(x_val, y_val) 추가

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
                     self.w -= self.lr * w_grad         # 가중치 업데이트
                     self.b -= b_grad                   # 절편 업데이트
                     # 가중치를 기록합니다.
                     self.w_history.append(self.w.copy())
                     # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적합니다
                     a = np.clip(a, 1e-10, 1-1e-10)
                     loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
                 # 에포크마다 평균 손실을 저장합니다
                 self.losses.append(loss/len(y))
                 # 검증 세트에 대한 손실을 계산합니다
                 self.update_val_loss(x_val, y_val)
     ```

     <br>

  3. 검증 손실 계산하기

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

     * update_val_loss() 메서드는 검증 세트의 손실을 계산하는 메서드로 fit() 메서드에서 손실을 계산하는 방식과 동일함
     * 검증 세트 샘플을 정방향으로 계산한 다음 활성화 함수를 통과시켜 출력값을 계산하고 이를 사용하여 로지스틱 손실 함수의 값을 계산해서 val_losses 리스트에 추가함

  4. 모델 훈련하기

     ```python
     layer3 = SingleLayer()
     layer3.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val)
     ```

  5. 손실값으로 그래프 그려 에포크 횟수 지정하기

     ```python
     plt.ylim(0, 0.3)
     plt.plot(layer3.losses)
     plt.plot(layer3.val_losses)
     plt.ylabel('loss')
     plt.xlabel('epoch')
     plt.legend(['train_loss', 'val_loss'])
     plt.show()
     ```

     ![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section2/github/image04.PNG?raw=true)

     * 에포크마다 훈련 세트와 검증 세트의 손실값을 self.val_losses에 저장했고 이를 이용하여 그래프를 그리면 다음과 같음
     * 검증 손실이 대략 20번째 에포크 이후에 훈련 세트보다 높아짐(20번의 에포크 이후에는 훈련할 필요가 없음)
     * 에포크가 진행됨에 따라 가중치는 훈련 세트에 잘 맞게 되지만 검증 세트에는 잘 맞지 않게 됨

  6. 훈련 조기 종료하기

     ```python
     layer4 = SingleLayer()
     layer4.fit(x_train_scaled, y_train, epochs=20)
     layer4.score(x_val_scaled, y_val)
     ```

     * 조기 종료(early stopping): 훈련을 일찍 멈추는 기법