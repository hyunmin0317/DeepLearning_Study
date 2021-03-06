# chap 03-3 손실 함수와 경사 하강법의 관계를 알아봅니다

2021.03.15

```markdown
* 경사 하강법: 어떤 손실 함수가 정의되었을 때 손실 함수의 값이 최소가 되는 지점을 찾아가는 방법
* 손실 함수(loss function): 예상한 값과 실제 타깃값의 차이를 정의한 함수
```

<br>

### 01. 손실 함수의 정체를 파헤쳐봅니다

* 제곱 오차(squared error): 타깃값과 예측값을 뺀 다음 제곱한 손실함수, SE = (y-ŷ)**2
  * 제곱 오차가 최소면 산점도 그래프를 가장 잘 표현한 직선이 그려짐
  * 제곱 오차의 최솟값을 찾는 방법을 통해 모델을 쉽게 만들 수 있음

* 제곱 오차를 가중치나 절편에 대해 미분하여 구한 기울기에 따라 함수값을 낮은 쪽으로 이동해 최솟값을 구함

* 가중치에 대하여 제곱 오차 미분하기

  * 제곱 오차를 가중치에 대하여 미분한 결과 (w에 대해 편미분)

    ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap03/section03/image01.PNG?raw=true)

  * 깔끔하게 표현하기 위해 보통은 제곱 오차 공식을 2로 나눈 함수를 편미분하여 (-(y-ŷ)x)을 사용

  * 가중치 업데이트 (여기서는 손실함수의 낮은 쪽으로 이동하기 위해 w에서 변화율을 뺌)

    ![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap03/section03/image02.PNG?raw=true)

  * 앞에서 오차 역전파를 알아보며 적용했던 수식과 같음 (w  + w_rate * err)

    ```python
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    ```

  <br>

* 절편에 대하여 제곱 오차 미분하기

  * 제곱 오차를 절편에 대하여 미분한 결과 (1/2을 곱한 제곱 오차 공식 사용)

    ![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap03/section03/image03.PNG?raw=true)

  * 절편 업데이트 (여기서는 손실함수의 낮은 쪽으로 이동하기 위해 b에서 변화율을 뺌)

    ![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap03/section03/image04.PNG?raw=true)

  * 앞에서 오차 역전파를 알아보며 적용했던 수식과 같음 (b  + 1 * err)

    ```python
    err = y_i - y_hat
    b= b + 1 * err
    ```

* 손실 함수 변화율의 값을 계산하기 위해 편미분을 사용함 

* 인공지능 분야에서 변화율을 그레이디언트(gradient, 경사)라고 부름