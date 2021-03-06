# chap 04-3 로지스틱 손실 함수를 경사 하강법에 적용합니다

2021.03.25

``` markdown
* 로지스틱 회귀의 목표: 올바르게 분류된 샘플 데이터의 비율 자체를 높이는 것이 분류의 목표
분류의 정확도는 미분 가능한 함수가 아니므로 경사 하강법의 손실 함수가 아닌 로지스틱 손실 함수를 사용
```

<br>

### 01. 로지스틱 손실 함수를 제대로 알아봅시다

* 로지스틱 손실 함수: 다중 분류를 위한 손실 함수인 크로스 엔트로피(cross entropy) 손실 함수를 이진 분류 버전으로 만든 함수
  * L = - (ylog(a)+(1-y)log(1-a))	(a=활성화 함수가 출력한 값, y=타깃)
  * 타깃의 값 (이진 분류이므로 1 또는 0)
    * y가 1인 경우 (양성 클래스): -log(a)
    * y가 0인 경우 (음성 클래스): -log(1-a)
  * 손실 함수를 최소로 만들기 위해서는 a의 값이 바뀌어야함 (양성 클래스는 a가 1에 음성 클래스는 a가 0에 가까워짐)
  * 로지스틱 손실 함수의 최솟값을 만드는 가중치와 절편을 찾기 위해 미분 

<br>

### 02. 로지스틱 손실 함수 미분하기

* 제곱 오차와 로지스틱 손실 함수의 미분

  ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section03/image01.PNG?raw=true)

  * 로지스틱 회귀의 구현은 3장에서 만든 Neuron 클래스와 크게 다르지 않음

* 로지스틱 손실 함수의 미분 (로지스틱 손실 함수의 값을 최소로 하는 가중치와 절편을 찾기 위함)

  1. 로지스틱 손실 함수와 연쇄 법칙

     ![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section03/image02.PNG?raw=true)

  2. 로지스틱 손실 함수를 a에 대하여 미분하기

     ![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section03/image03.PNG?raw=true)

  3. a를 z에 대하여 미분하기

     ![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section03/image04.PNG?raw=true)

  4. z를 w에 대하여 미분하기

     ![image05](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section03/image05.PNG?raw=true)

  5. 로지스틱 손실 함수를 w에 대하여 미분하기

     ![image06](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section03/image06.PNG?raw=true)

<br>

### 03. 로지스틱 손실 함수의 미분 과정 정리하고 역전파 이해하기

![image07](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section03/image07.PNG?raw=true)

* 로지스틱 손실 함수 L은 a에 대해, a는 z에 대해, z는 w에 대해 미분하고 각 도함수의 곱을 가중치 업데이트에 사용

* 로지스틱 손실 함수에 대한 미분이 연쇄 법칙에 의해 진행되는 구조를 보고 ''그레디언트가 역전파된다''고 함

* 가중치와 절편 업데이트 방법

  1. 가중치 업데이트 방법 정리하기

     ![image08](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section03/image08.PNG?raw=true)

     * 로지스틱 손실 함수를 가중치에 대해 미분한 식을 가중치에서 빼는 방법으로 가중치 업데이트

  2. 절편 업데이트 방법 정리하기

     ![image09](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section03/image09.PNG?raw=true)

     * 로지스틱 손실 함수를 절편에 대해 미분한 식을 절편에서 빼는 방법으로 가중치 업데이트