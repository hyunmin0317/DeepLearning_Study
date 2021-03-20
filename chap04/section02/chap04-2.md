# chap 04-2 시그모이드 함수로 확률을 만듭니다

2021.03.20

<br>

### 01. 시그모이드 함수의 역할을 알아봅니다

* 로지스틱 회귀의 전체 구조

  ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section02/image01.PNG?raw=true)

  * 가장 왼쪽에 있는 뉴런은 선형 함수이며 함수의 출력값 z는 다음과 같음

    ![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section02/image02.PNG?raw=true)

  * 로지스틱 회귀의 시그모이드 함수(활성화 함수)는 출력값 z를 0~1 사이의 확률값 a로 변환하는 역할을 함

  * 보통 시그모이드 함수를 통과한 값 확률 a가 0.5(50%)보다 크면 양성 클래스, 그 이하면 음성 클래스로 구분함

<br>

### 02. 시그모이드 함수가 만들어지는 과정을 살펴봅니다

* 시그모이드 함수가 만들어지는 과정: 오즈 비 > 로짓 함수 > 시그모이드 함수

  * 오즈 비에 대해 알아볼까요?

    * 오즈 비(odds ratio): 성공 확률과 실패 확률의 비율을 나타내는 통계이며 시그모이드 함수의 기반이 됨

    * 오즈 비의 값은 p가 0부터 1까지 증가할 때 처음에는 천천히 증가하지만 p가 1에 가까워질수록 급격하게 증가함

      ![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section02/image03.PNG?raw=true)

  * 로짓 함수에 대해 알아볼까요?

    * 로짓 함수(logit function): 오즈 비에 로그 함수를 취하여 만든 함수

      ![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section02/image04.PNG?raw=true)

      * p가 0.5일 때 0이 되고 p가 0과 1일 때 각각 무한대로 음수와 양수가 되는 특징을 갖음
    
      * 로짓 함수의 세로 축을 z로 가로 축을 p로 놓으면 확률 p가 0에서 1까지 변할 때 z가 크게 변하며 식은 다음과 같음
      
        ![image05](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section02/image05.PNG?raw=true)
  
  * 로지스틱 함수에 대해 알아볼까요?
  
    * 로지스틱 함수(시그모이드 함수): 로짓 함수를 가로 축을 z로 놓기 위해 z에 대해 정리한 식
  
      ![image06](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section02/image06.PNG?raw=true)
  
      * 로지스틱 함수의 모양은 S자형으로 모양에서 착안하여 로지스틱 함수를 시그모이드 함수(sigmoid function)라고 부름

<br>

### 03. 로지스틱 회귀 중간 정리하기

![image07](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section02/image07.PNG?raw=true)

* 로지스틱 회귀의 과정
  1. 뉴런은 선형 함수이며 함수의 출력값 z의 범위는 -∞부터 ∞로 z의 값을 조절할 방법이 필요함
  2. z의 값을 활성화 함수인 시그모이드 함수를 사용하고 z를 확률로 해석하여 범위가 0부터 1인 확률 a를 출력함
  3. 시그모이드 함수의 확률인 a를 0과 1로 구분하기 위해서 임계 함수를 사용하여 입력 데이터를 0 또는 1의 값으로 나눔(이진분류)
* 로지스틱 회귀는 이진분류 알고리즘이며 가중치와 절편을 적절하게 업데이트할 수 있는 방법인 로지스틱 손실 함수를 앞으로 배움