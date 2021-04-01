# chap 09-1 순차 데이터와 순환 신경망을 배웁니다

2021.04.02

`지금까지 인공신경망에 사용한 데이터는 각 샘플이 독립적이라고 가정하고 에포크마다 전체 샘플을 섞은 후에 모델 훈련을 진행함`

<br>

### 01. 순차 데이터를 소개합니다

<img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image01.PNG?raw=true" alt="image01.PNG" style="zoom:50%;" />

<br>

### 02. 순환 신경망을 소개합니다

<img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image02.PNG?raw=true" alt="image02.PNG" style="zoom:50%;" />

* 순환 신경망은 뉴런을 셀이라 부릅니다

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image03.PNG?raw=true" alt="image03.PNG" style="zoom:80%;" />

* 순환층의 셀에서 수행되는 계산 과정

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image04.PNG?raw=true" alt="image04.PNG" style="zoom: 80%;" />

<br>

### 03. 순환 신경망의 정방향 계산을 알아봅니다

<img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image05.PNG?raw=true" alt="image05.PNG" style="zoom: 67%;" />

<br>

### 04. 순환 신경망의 역방향 계산을 알아봅니다

* 가중치 W2에 대한 손실 함수의 도함수를 구합니다

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image06.PNG?raw=true" alt="image06.PNG" style="zoom: 67%;" />

* Z1에 대한 H의 도함수를 구합니다

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image07.PNG?raw=true" alt="image07.PNG" style="zoom: 67%;" />

* 가중치 W1h에 대한 Z1의 도함수를 구합니다

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image08.PNG?raw=true" alt="image08.PNG" style="zoom: 67%;" />

  * 현재 타임 스텝의 은닉 상태와 이전 타임 스텝의 은닉 상태를 포함한 순환 신경망의 계산 과정

    <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image09.PNG?raw=true" alt="image09.PNG" style="zoom: 67%;" />

  * 미분의 곱셈 법칙을 사용하여 도함수를 다시 계산

    <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image10.PNG?raw=true" alt="image10.PNG" style="zoom: 67%;" />

* 가중치 W1h, W1x, b1에 대한 Z1의 도함수

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section1/image11.PNG?raw=true" alt="image11.PNG" style="zoom: 67%;" />

