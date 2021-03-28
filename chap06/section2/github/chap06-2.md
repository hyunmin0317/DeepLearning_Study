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

<br>

### 03. 은닉층이 추가된 신경망을 알아봅니다

![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image03.PNG?raw=true)

<br>

### 04. 다층 신경망의 개념을 정리합니다

![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image04.PNG?raw=true)

<br>

### 05. 다층 신경망에 경사 하강법을 적용합니다

![image05](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image05.PNG?raw=true)



* 가중치에 대하여 손실 함수를 미분합니다(출력층)

  ![image06](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image06.PNG?raw=true)

* 절편에 대하여 손실 함수를 미분합니다(출력층)

  ![image07](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image07.PNG?raw=true)

* 가중치에 대하여 손실 함수를 미분합니다(은닉층)

  ![image08](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image08.PNG?raw=true)

* 도함수를 곱합니다(은닉층)

  ![image09](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image09.PNG?raw=true)

![image10](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image10.PNG?raw=true)

* 절편에 대하여 손실 함수를 미분하고 도함수를 곱합니다

  ![image11](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap06/section2/github/image11.PNG?raw=true)

### 06. 2개의 층을 가진 신경망 구현하기

<br>

### 07. 모델 훈련하기

<br>

### 08. 가중치 초기화 개선하기