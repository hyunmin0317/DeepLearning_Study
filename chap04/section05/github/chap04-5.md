# chap 04-5 로지스틱 회귀를 위한 뉴런을 만듭니다

2021.03.25

```markdown
모델을 만들기 전에 성능 평가 방법에 대해 잠시 알아보겠습니다.
```

<br>

### 01. 모델의 성능 평가를 위한 훈련 세트와 데이터 세트

* 일반화 성능(generalization performeance): 훈련된 모델의 실전 성능
* 올바르게 모델의 성능을 측정하기 위해 훈련 데이터 세트를 두 덩어리로 나누어 하나는 훈련에, 다른 하나는 테스트에 사용
* 훈련 데이터 세트를 훈련 세트와 테스트 세트로 나누는 규칙
  * 훈련 데이터 세트를 나눌 때는 테스트 세트보다 훈련 세트가 더 많아야 함
  * 훈련 데이터 세트를 나누기 전에 양성, 음성 크래스가 훈련 세트나 테스트 세트의 어느 한쪽에 몰리지 않도록 골고루 섞어야 함

<br>

### 02. 훈련 세트와 테스트 세트로 나누기

* 양성, 음성 클래스가 훈련 세트와 테스트 세트에 고르게 분포하도록 만들어야 함

* 양성, 음성 클래스의 비율을 일정하게 유지하면서 훈련 데이터 세트를 훈련 세트와 테스트 세트로 나누기

  1. train_test_split() 함수로 훈련 데이터 세트 나누기

     * sklear.model_selection 모듈에서 train_test_split() 함수를 import

       ```python
       from sklearn.model_selection import train_test_split
       ```
       * train_test_split() 함수: 입력된 훈련 데이터 세트를 훈련 세트 75%, 테스트 세트 25%의 비율로 나눠주는 함수

     * train_test_split() 함수에 입력 데이터 x, 타깃 데이터 y와 설정을 매개변수로 지정

       ```python
       x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)
       ```

       * 매개변수 설정
         1. stratify=y : 훈련 데이터를 나눌 때 클래스 비율을 동일하게 만들기 위해 사용
            * 일부 클래스 비율이 불균형한 경우에는 stratify를 y로 지정
         2. test_size=0.2 :  훈련 데이터 세트의 비율을 조절하고 싶을 때 사용함
            * 입력된 데이터 세트의 20%를 테스트 세트로 나누기 위해 test_size에 0.2를 전달
         3. random_state=42 : 난수 초깃값을 지정하기 위해 사용
            * 데이터 세트를 섞은 다음 나눈 결과가 항상 일정하도록 random_state 매개변수에 난수 초깃값 42를 지정

     <br>

  2. 결과 확인하기

     ```python
     print(x_train.shape, x_test.shape)	# (455, 30) (114, 30)
     ```

     * 훈련 데이터 세트가 잘 나누어졌는지 훈련 세트와 테스트 세트 비율을 shape 속성을 이용해 확인 (4:1의 비율로 잘 나누어짐)

  3. unique() 함수로 훈련 세트의 타깃 확인하기

     ```python
     np.unique(y_train, return_counts=True)	# (array([0, 1]), array([170, 285]))
     ```

     * 넘파이의 unique() 함수로 훈련 세트의 타깃 안에 있는 클래스의 개수 확인 (클래스의 비율이 그대로 유지됨)

<br>

### 03. 로지스틱 회귀 구현하기

* 정방향으로 데이터가 흘러가는 과정(정방향 계산)과 가중치를 업데이트하기 위해 역방향으로 데이터가 흘러가는 과정(역방향 계산)을 구현해야 함

* LogisticNeuron 클래스

  ```python
  class LogisticNeuron:
      
      def __init__(self):
      	self.w = None
      	self.b = None
      
      def forpass(self, x):
      	z = np.sum(x * self.w) + self.b		# 직선 방정식을 계산합니다.
      	return z
  
  	def backprop(self, x, err):
      	w_grad = x * err	# 가중치에 대한 그레이디언트 계산
      	b_grad = 1 * err	# 절편에 대한 그레이디언트 계산
      	return w_grad, b_grad
  ```

  * __init()__ 메서드는 가중치와 절편을 미리 초기화하지 않습니다
    * 입력 데이터의 특성이 많아 가중치를 미리 초기화하지 않고 특성 개수에 맞게 결정
  * forpass() 메서드에 넘파이 함수를 사용합니다
    * 가중치와 입력 특성의 곱을 모두 더하기 위해 np.sum() 함수를 사용

<br>

### 04. 훈련하는 메서드 구현하기

<br>

### 05. 예측하는 메서드 구현하기

<br>

### 06. 구현 내용 한눈에 보기

<br>

### 07. 로지스틱 회귀 모델 훈련시키기





