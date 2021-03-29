# chap 06-2 2개의 층을 가진 신경망을 구현합니다

2021.03.28

<br>

### 01. 하나의 층에 여러 개의 뉴런을 사용합니다

![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image01.PNG?raw=true)

* 하나의 층에 여러 개의 뉴런을 사용하면 입력층에서 전달되는 특성이 각 뉴런에 모두 전달됨
* 그림에서 3개의 특성은 각각 2개의 뉴런에 모두 전달되어 z1, z2를 출력하고 계산식과 행렬식으로 표현하면 다음과 같음

<br>

### 02. 출력을 하나로 모읍니다

![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image02.PNG?raw=true)

* 각 뉴런에서 출력된 값(z1, z2, ...)을 하나의 뉴런으로 다시 모아야 함
* 출력된 값을 활성화 함수에 통과시킨 값(활성화 출력)이 마지막 뉴런에 입력되고 여기에 절편이 더해져 z가 만들어짐 

<br>

### 03. 은닉층이 추가된 신경망을 알아봅니다

![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image03.PNG?raw=true)

* 전체 구조는 다음과 같으며 2개의 뉴런과 2개의 층을 가진 신경망으로 구성되어 있음
* 구성 요소
  * 입력층: 입력값이 모여 있는 층으로 보통 층의 개수에 포함시키지 않음
  * 은닉층: 입력층의 값들이 출력층으로 전달되기 전에 통과하는 단계로 2개의 뉴런으로 구성되어 있음
  * 출력층: 활성화 출력을 입력받고 절편을 더해 결과값 z를 출력함

<br>

### 04. 다층 신경망의 개념을 정리합니다

![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image04.PNG?raw=true)

<br>

* 그림에서 n개의 입력이 m개의 뉴런으로 입력되고 은닉층을 통과한 값들은 다시 출력층으로 모이며 이를 딥러닝이라고 부름
* 다층 신경망에서 알아야 할 주의 사항과 개념
  * 활성화 함수는 층마다 다를 수 있지만 한 층에서는 같아야 합니다
    * 은닉층과 출력층에 있는 모든 뉴런에는 활성화 함수가 필요하며 문제에 맞는 활성화 함수를 사용해야 함
  * 모든 뉴런이 연결되어 있으면 완전 연결(fully-connected) 신경망이라고 합니다
    * 완전 연결 신경망은 인공신경망의 한 종류이며, 가장 기본적인 신경망 구조
    * 완전 연결층: 뉴런이 모두 연결되어 있는 층

### 05. 다층 신경망에 경사 하강법을 적용합니다

![image05](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image05.PNG?raw=true)

* 다층 신경망 예측 과정

  1. 입력 데이터 X와 가중치 W1을 곱하고 절편 b1은 더해 Z1이 되고 활성화 함수를 통과하여 A1이 됨 (첫 번째 은닉층)
  2. 활성화 출력 A1과 출력층의 가중치 W2를 곱하고 절편 b2를 더해 Z2를 만들고 활성화 함수를 통과하여 A2가 됨 (출력층)
  3. A2의 값을 보고 0.5보다 크면 양성, 그렇지 않으면 음성으로 예측 (결과값 Y)

* 경사 하강법을 적용하려면 각 층의 가중치와 절편에 대한 손실함수 L의 도함수를 구해야 함

  <br>

* 신경망에 경사 하강법을 적용하기 위해 미분하는 과정 (출력층에서 은닉층 방향으로 미분)
  * 가중치에 대하여 손실 함수를 미분합니다(출력층)

    ![image06](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image06.PNG?raw=true)

  * 절편에 대하여 손실 함수를 미분합니다(출력층)

    ​	![image07](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image07.PNG?raw=true)

  * 가중치에 대하여 손실 함수를 미분합니다(은닉층)

    ![image08](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image08.PNG?raw=true)

  * 도함수를 곱합니다(은닉층)

    ![image09](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image09.PNG?raw=true)

  * 오차 그레이디언트를 W1에 적용하는 방법

  ![image10](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image10.PNG?raw=true)

  * 절편에 대하여 손실 함수를 미분하고 도함수를 곱합니다

    ![image11](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image11.PNG?raw=true)

### 06. 2개의 층을 가진 신경망 구현하기

`SingleLayer 클래스를 상속하여 DualLayer 클래스를 만들고 필요한 메서드만 재정의`

1. SingleLayer 클래스를 상속한 DualLayer 클래스 만들기

   ```python
   class DualLayer(SingleLayer):
       
       def __init__(self, units=10, learning_rate=0.1, l1=0, l2=0):
           self.units = units         # 은닉층의 뉴런 개수
           self.w1 = None             # 은닉층의 가중치
           self.b1 = None             # 은닉층의 절편
           self.w2 = None             # 출력층의 가중치
           self.b2 = None             # 출력층의 절편
           self.a1 = None             # 은닉층의 활성화 출력
           self.losses = []           # 훈련 손실
           self.val_losses = []       # 검증 손실
           self.lr = learning_rate    # 학습률
           self.l1 = l1               # L1 손실 하이퍼파라미터
           self.l2 = l2               # L2 손실 하이퍼파라미터
   ```

   * 은닉층의 뉴런 개수를 지정하는 units 매개변수 추가
   * 은닉층과 출력층의 가중치와 절편을 각각 w1, b1과 w2, b2에 저장

2. forpass() 메서드 수정하기

   ```python
   def forpass(self, x):
           z1 = np.dot(x, self.w1) + self.b1        # 첫 번째 층의 선형 식을 계산합니다
           self.a1 = self.activation(z1)            # 활성화 함수를 적용합니다
           z2 = np.dot(self.a1, self.w2) + self.b2  # 두 번째 층의 선형 식을 계산합니다.
           return z2
   ```

   * 은닉층과 출력층의 정방향 계산을 수행 (활성화 함수를 통과한 a1과 출력층의 가중치 w2를 곱하고 b2를 더해 결과값 z2를 반환)

3. backprop() 메서드 수정하기

   ```python
   def backprop(self, x, err):
           m = len(x)       # 샘플 개수
           # 출력층의 가중치와 절편에 대한 그래디언트를 계산합니다.
           w2_grad = np.dot(self.a1.T, err) / m
           b2_grad = np.sum(err) / m
           # 시그모이드 함수까지 그래디언트를 계산합니다.
           err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1 - self.a1)
           # 은닉층의 가중치와 절편에 대한 그래디언트를 계산합니다.
           w1_grad = np.dot(x.T, err_to_hidden) / m
           b1_grad = np.sum(err_to_hidden, axis=0) / m
           return w1_grad, b1_grad, w2_grad, b2_grad
   ```

   * 출력층의 가중치(w2_grad)와 절편(b2_grad)의 계산 공식

     ![image14](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image14.PNG?raw=true)

   * 은닉층의 가중치(w1_grad)와 절편(b1_grad)의 계산 공식

     ![image15](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image15.PNG?raw=true)

4. fit() 메서드 수정하기

   * 은닉층과 출력층의 가중치, 절편을 초기화하고 에포크마다 정방향 계산을 수행하여 오차 계산하며 오차를 역전파하여 가중치와 절편의 그레디언트를 계산하고 손실을 계산하여 누적하는 역할
   * fit() 메서드를 3개의 작은 메서드로 나누어 구현

5. fit() 메서드의 가중치 초기화 부분을 init_weights() 메서드로 분리

   ```python
   def init_weights(self, n_features):
           self.w1 = np.ones((n_features, self.units))  # (특성 개수, 은닉층의 크기)
           self.b1 = np.zeros(self.units)               # 은닉층의 크기
           self.w2 = np.ones((self.units, 1))           # (은닉층의 크기, 1)
           self.b2 = 0
   ```

   * 입력 특성의 개수를 지정하는 n_features 매개변수 하나를 갖음

6. fit() 메서드의 for문 안에 일부 코드를 training() 메서드로 분리

   ```python
   def fit(self, x, y, epochs=100, x_val=None, y_val=None):
           y = y.reshape(-1, 1)          # 타깃을 열 벡터로 바꿉니다.
           y_val = y_val.reshape(-1, 1)
           m = len(x)                    # 샘플 개수를 저장합니다.
           self.init_weights(x.shape[1]) # 은닉층과 출력층의 가중치를 초기화합니다.
           # epochs만큼 반복합니다.
           for i in range(epochs):
               a = self.training(x, y, m)
               # 안전한 로그 계산을 위해 클리핑합니다.
               a = np.clip(a, 1e-10, 1-1e-10)
               # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
               loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
               self.losses.append((loss + self.reg_loss()) / m)
               # 검증 세트에 대한 손실을 계산합니다.
               self.update_val_loss(x_val, y_val)
               
       def training(self, x, y, m):
           z = self.forpass(x)       # 정방향 계산을 수행합니다.
           a = self.activation(z)    # 활성화 함수를 적용합니다.
           err = -(y - a)            # 오차를 계산합니다.
           # 오차를 역전파하여 그래디언트를 계산합니다.
           w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)
           # 그래디언트에 페널티 항의 미분 값을 더합니다
           w1_grad += (self.l1 * np.sign(self.w1) + self.l2 * self.w1) / m
           w2_grad += (self.l1 * np.sign(self.w2) + self.l2 * self.w2) / m
           # 은닉층의 가중치와 절편을 업데이트합니다.
           self.w1 -= self.lr * w1_grad
           self.b1 -= self.lr * b1_grad
           # 출력층의 가중치와 절편을 업데이트합니다.
           self.w2 -= self.lr * w2_grad
           self.b2 -= self.lr * b2_grad
           return a
   ```

   * 정방향 계산과 그레이디언트를 업데이트하는 코드를 training() 메서드로 옮김
   * fit() 메서드는 훈련 데이터 x, y와 훈련 샘플의 개수 m을 매개변수로 받고 마지막 출력층의 활성화 출력 a를 반환

7. reg_loss() 메서드 수정하기

   ```python
   def reg_loss(self):
           # 은닉층과 출력층의 가중치에 규제를 적용합니다.
           return self.l1 * (np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + self.l2 / 2 * (np.sum(self.w1**2) + np.sum(self.w2**2))
   ```

   * 은닉층과 출력층의 가중치에 대한 L1, L2 손실을 계산

<br>

### 07. 모델 훈련하기

1. 다층 신경망 모델 훈련하고 평가하기

   ```python
   dual_layer = DualLayer(l2=0.01)
   dual_layer.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val, epochs=20000)
   dual_layer.score(x_val_scaled, y_val)
   ```

   * L2 규제는 0.01만큼, 에포크는 20000번으로 지정하여 다층 신경망 모델을 훈련하고 모델 평가 (문제가 간단하여 평가점수 동일)

2. 훈련 손실과 검증 손실 그래프 분석하기

   ```python
   plt.ylim(0, 0.3)
   plt.plot(dual_layer.losses)
   plt.plot(dual_layer.val_losses)
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train_loss', 'val_loss'])
   plt.show()
   ```

   ![image12](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image12.PNG?raw=true)
   
   * 훈련 손실 그래프는 훈련 데이터로 손실 함수의 최솟값을 찾아가는 과정을 보여주고 검증 손실 그래프는 검증 데이터로 손실 함수의 최솟값을 찾아가는 과정을 보여줌
   * `SingleLayer` 클래스보다 가중치의 개수가 훨씬 많아져 학습하는 데 시간이 오래 걸리기 때문에 손실 그래프가 천천히 감소

<br>

### 08. 가중치 초기화 개선하기

* 손실 함수가 감소하는 방향을 올바르게 찾는 데 시간이 많이 소요되어 손실 그래프의 초기 손실값이 감소하는 곡선이 매끄럽지 않음
* 이는 가중치 초기화와 관련이 깊으며 기존은 가중치를 1로 놓았고 이를 개선하기 위해 random.normal() 함수를 통해 가중치 초기화

* 가중치 초기화 개선 과정

  1. 가중치 초기화를 위한 init_weights() 메서드 수정하기

     ```python
     class RandomInitNetwork(DualLayer):
         
         def init_weights(self, n_features):
             np.random.seed(42)
             self.w1 = np.random.normal(0, 1, (n_features, self.units))  # (특성 개수, 은닉층의 크기)
             self.b1 = np.zeros(self.units)                        		# 은닉층의 크기
             self.w2 = np.random.normal(0, 1, (self.units, 1))           # (은닉층의 크기, 1)
             self.b2 = 0
     ```

     * 실행 결과를 동일하게 하기 위해 np.random.seed() 함수를 통해 무작위 수의 초깃값 고정 (실전에서는 필요 없음)
     * normal() 함수의 매개변수는 순서대로 평균, 표준 편차, 배열 크기

  2. RandomInitNetwork 클래스 객체를 다시 만들고 모델 훈련

     ```python
     random_init_net = RandomInitNetwork(l2=0.01)
     random_init_net.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val, epochs=500)
     
     plt.plot(random_init_net.losses)
     plt.plot(random_init_net.val_losses)
     plt.ylabel('loss')
     plt.xlabel('epoch')
     plt.legend(['train_loss', 'val_loss'])
     plt.show()
     ```

     ![image13](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image13.PNG?raw=true)

     * 가중치를 모두 1로 초기화한 것보다 무작위 수로 초기화한 것이 학습 성능에 영향을 미쳐 손실 함수가 감소하는 곡선이 매끄럽고 손실 함수 값이 훨씬 빠르게 줄어든 것을 확인할 수 있음