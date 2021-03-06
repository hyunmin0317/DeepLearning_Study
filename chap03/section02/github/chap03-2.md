# chap 03-2 경사 하강법으로 학습하는 방법을 알아봅니다

2021.03.14

<br>

### 01. 그래프로 경사 하강법의 의미를 알아봅니다

```markdown
 * 여러 개의 특성을 가진 데이터를 이용하여 그래프를 그릴 때 알고리즘에 대한 직관을 쉽게 얻고 낮은 차원에서 얻은 직관은 높은 차원으로 확장될 수 있으므로 입력 데이터의 특성 1개를 골라 시각화하는 경우가 많음
```

* 선형 회귀와 경사 하강법의 관계를 이해합시다
  * 선형 회귀의 목표: 입력 데이터(x)와 타깃 데이터(y)를 통해 기울기(a)와 절편(b)을 찾아 산점도 그래프를 잘 표현하는 직선의 방정식을 구하는 것이 목표
  * 회귀 문제를 푸는 알고리즘
    * 경사 하강법(gradient descent): 모델이 데이터를 잘 표현할 수 있도록 기울기(변화율)를 사용하여 모델을 조금씩 조정하는 최적화 알고리즘
    * 정규 방정식(Normal Equation), 결정 트리(Decision Tree), 서포트 벡터 머신(Support Vector Machine)

<br>

### 02. 예측값과 변화율에 대해 알아봅니다

```markdown
* 딥러닝 분야에서 모델 표현법(직선의 방정식): ŷ=wx+b (w=가중치, b=절편, ŷ=예측값)
```

* 예측값이란 무엇일까요?
  * 예측값: 입력과 출력 데이터(x,y)를 통해 규칙(a,b)을 발견하여 만든 모델에 새로운 입력값을 넣어 나온 출력 (모델을 통해 예측한 값)

<br>

### 03. 예측값으로 올바른 모델 찾기

* 훈련 데이터(x,y)에 잘 맞는 w와 b를 찾는 방법
  1. 무작위로 w와 b를 정합니다. (무작위로 모델 만들기)
  2. x에서 샘플 하나를 선택하여  ŷ을 계산합니다. (무작위로 모델 예측하기)
  3.  ŷ과 선택한 샘플의 진짜 y를 비교합니다. (예측한 값과 진짜 정답 비교하기, 틀릴 확률 99%)
  4.  ŷ이 y와 더 가까워지도록 w, b를 조정합니다. (모델 조정하기)
  5. 모든 샘플을 처리할 때까지 다시 2~4 항목을 반복합니다.
* 훈련 데이터에 맞는 w와 b 찾아보기 (예제)
  1. w와 b 초기화하기

     * w와 v를 무작위로 초기화 (예제에서는 간단하게 두 값을 모두 실수 1.0으로 정함)

       ```python
       w = 1.0
       b = 1.0
       ```

  2. 훈련 데이터의 첫 번째 샘플 데이터로  ŷ 얻기

     * 임시로 만든 모델로 훈련 데이터의 첫 번째 샘플 x[0]에 대한 ŷ을 계산하여 y_hat 변수에 저장

       ```python
       y_hat = x[0] * w + b
       print(y_hat)	# 1.0616962065186886
       ```

  3. 타깃과 예측 데이터 비교하기

     * 첫 번째 샘플 x[0]에 대응하는 타깃값 y[0]을 출력하여 y_hat의 값과 비교

       ```python
       print(y[0])	# 151.0
       ```

  4. w 값 조절해 예측값 바꾸기

     * w와 b를 무작위 값으로 정하여 예측 결과가 잘 나오지 않아 w와 b를 좀 더 좋은 방향으로 바꿈

     * w와 b를 조금씩 변경해서 y_hat의 변화량을 관찰하여 y_hat이 y[0]에 가까워질 수 있도록 바꿈

       ```python
       w_inc = w + 0.1		# w를 0.1만큼 증가시키고 y_hat의 변화량 관찰
       y_hat_inc = x[0] * w_inc + b
       print(y_hat_inc)	# 1.0678658271705574
       # w 값을 0.1만큼 증가시킨 다음 값을 다시 예측하여 y_hat_inc에 저장 (y_hat보다 증가)
       ```

  5. w 값 조정한 후 예측값 증가 정도 확인하기

     ```python
     w_rate = (y_hat_inc - y_hat) / (w_inc - w)
     print(w_rate)	# 0.061696206518688734
     # y_hat이 증가한 양을 w가 증가한 양으로 나누어 w가 0.1만큼 증가했을 때 y_hat의 증가량 계산
     ```
  
     * 변화율에 따라 w 값을 조정하여 올바른 모델을 찾음

<br>

### 04. 변화율로 가중치(w) 업데이트하기 (y_hat 증가시키는 상황)

* 변화율이 양수일 때 가중치를 업데이트하는 방법

  * w가 증가하면 y_hat도 증가 (변화율(양수)을 w에 더하면 w와 y_hat 증가)

* 변화율이 음수일 때 가중치를 업데이트하는 방법

  * w가 증가하면 y_hat은 감소 (변화율(음수)을 w에 더하면 w와 y_hat 증가)

* 두 경우 모두 변화율을 w에 더하면 y_hat 증가 (가중치 w를 업데이트하는 방법: w + w_rate)

  ```python
  w_new = w + w_rate
  print(w_new)	# 1.0616962065186888
  ```

<br>

### 05. 변화율로 절편(b) 업데이트하기 (y_hat 증가시키는 상황)

* 절편 b에 대한 변화율로 b를 업데이트 (b를 0.1만큼 증가시킨 수 y_hat의 변화율 계산)

  ```
  b_inc = b + 0.1
  y_hat_inc = x[0] * w + b_inc
  print(y_hat_inc)	# 1.1616962065186887
  
  b_rate = (y_hat_inc - y_hat) / (b_inc - b)
  print(b_rate)	# 1.0
  ```

* b를 업데이트하기 위해서는 변화율이 1이므로 단순하게 1을 더하면 됨

  ```python
  b_new = b + 1
  print(b_new)	# 2.0
  ```

* ##### y_hat을 보고 w와 b를 업데이트하는 방법의 문제점 (수동적인 방법)

  * y_hat이 y에 한참 미치지 못하는 값인 경우 w와 b를 더 큰 폭으로 수정할 수 없음 (기준을 정하기 어려움)
  * y_hat이 y보다 커지면 y_hat을 감소시키지 못 함

<br>

### 06. 오차 역전파로 가중치와 절편을 더 적절하게 업데이트 (능동적인 방법)

* 오차 역전파(backpropagation): ŷ과 y의 차이를 이용하여 w와 b를 업데이트, 오차가 전파되는 모습으로 수행

* y에서 ŷ을 뺀 오차의 양을 변화율에 곱하는 방법으로 w를 업데이트

* 가중치와 절편을 더욱 적절하게 업데이트하는 방법

  1. 오차와 변화율을 곱하여 가중치 업데이트하기

     * x[0]일 때 w와 b의 변화율에 오차를 곱한 다음 업데이트된 w_new와 b_new를 출력

       ```python
       err = y[0] - y_hat
       w_new = w + w_rate * err
       b_new = b + 1 * err
       print(w_new, b_new)	# 10.250624555904514 150.9383037934813
       ```

  2. 다른 샘플로 오차와 변화율 새로 구하기

     * 두 번째 샘플 x[1]을 사용하여 오차를 구하고 새로운 w와 b를 구함

       ```python
       y_hat = x[1] * w_new + b_new	# w_rate와 샘플값이 같으므로 그대로 사용
       err = y[1] - y_hat
       w_rate = x[1]
       w_new = w_new + w_rate * err
       b_new = b_new + 1 * err
       print(w_new, b_new)	# 14.132317616381767 75.52764127612664
       # w는 4만큼 커지고 b는 절반으로 줄어듦
       ```

  3. 전체 샘플을 반복하기

     * 이 방식으로 모든 샘플을 사용해 가중치와 절편 업데이트

       ```python
       for x_i, y_i in zip(x, y): # 파이썬의 zip() 함수는 배열에서 동시에 요소를 하나씩 꺼내줌
           y_hat = x_i * w + b
           err = y_i - y_hat
           w_rate = x_i
           w = w + w_rate * err
           b = b + 1 * err
       print(w, b)	# 587.8654539985689	99.40935564531424
       ```

  4. 모델이 전체 데이터 세트를 잘 표현하는지 그래프를 그려 확인

     * 산점도 위에 w와 b를 사용한 직선을 그려 모델이 적합한지 확인

       ```python
       plt.scatter(x, y)
       pt1 = (-0.1, -0.1 * w + b)
       pt2 = (0.15, 0.15 * w + b)
       plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])	
       # 시작점과 종료점의 x, y 좌표를 plot() 함수에 전달하여 직선 그래프를 그림
       plt.xlabel('x')
       plt.ylabel('y')
       plt.show()
       ```

       ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap03/section02/github/image01.PNG?raw=true)

  5. 여러 에포크를 반복하기

     * 보통 경사 하강법에서는 주어진 훈련 데이터로 학습을 여러 번 반복하며 수천 번의 에포크를 반복함

     * 에포크(epoch): 전체 훈련 데이터를 모두 이용하여 진행하는 한 단위의 작업

     * 앞에서 찾은 모델에 100번의 에포크를 반복하며 모델 수정

       ```python
       for i in range(1, 100):
           for x_i, y_i in zip(x, y):
               y_hat = x_i * w + b
               err = y_i - y_hat
               w_rate = x_i
               w = w + w_rate * err
               b = b + 1 * err
       print(w, b)	# 913.5973364345905	123.39414383177204
       ```

     * 수정한 모델이 적합한지 확인

       ```python
       plt.scatter(x, y)
       plt1 = (-0.1, -0.1 * w + b)
       plt2 = (0.15, 0.15 * w + b)
       plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])	# ŷ = 913.6x + 123.4
       plt.xlabel('x')
       plt.ylabel('y')
       plt.show()
       ```

       ![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap03/section02/github/image02.PNG?raw=true)

  6. 모델로 예측하기

     * 입력 x에 없었던 새로운 데이터에(0.18) 대해 ŷ의 값 예측

       ```python
       x_new = 0.18
       y_pred = x_new * w + b
       print(y_pred)	# 287.8416643899983
       ```

     * 새로운 데이터를 산점도 위에 나타냄

       ```python
       plt.scatter(x, y)
       plt.scatter(x_new, y_pred)
       plt.xlabel('x')
       plt.ylabel('y')
       plt.show()
       ```

       ![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap03/section02/github/image03.PNG?raw=true)

<br>

### 07. 모델 만드는 방법 요약

1. w와 b를 임의의 값(1.0, 1.0)으로 초기화하고 훈련 데이터의 샘플을 하나씩 대입하여 y와 ŷ의 오차를 구합니다.
2. 구한 오차를 w와 b의 변화율에 곱하고 이 값을 이용하여 w와 b를 업데이트합니다.
3. 만약 ŷ이 y보다 커지면 오차는 음수가 되어 자동으로 w와 b가 줄어드는 방향으로 업데이트됩니다.
4. 반대로 ŷ이 y보다 작으면 오차는 양수가 되고 w와 b는 더 커지도록 업데이트 됩니다.