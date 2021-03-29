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
2. forpass() 메서드 수정하기
3. backprop() 메서드 수정하기
4. fit() 메서드 수정하기
5. fit() 메서드의 가중치 초기화 부분을 init_weights() 메서드로 분리
6. fit() 메서드의 for문 안에 일부 코드를 training() 메서드로 분리
7. reg_loss() 메서드 수정하기

<br>

### 07. 모델 훈련하기

1. 다층 신경망 모델 훈련하고 평가하기
2. 훈련 손실과 검증 손실 그래프 분석하기

<br>

### 08. 가중치 초기화 개선하기

1. 가중치 초기화를 위한 init_weights() 메서드 수정하기
2. RandomInitNetwork 클래스 객체를 다시 만들고 모델 훈련