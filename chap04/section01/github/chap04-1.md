# chap 04-1 초기 인공지능 알고리즘과 로지스틱 회귀를 알아봅니다

2021.03.

<br>

### 01. 퍼셉트론에 대해 알아봅니다

```markdown
이진 분류 문제에서 최적의 가중치를 학습하는 퍼셉트론(Perceptron) 알고리즘을 1957년에 프랑크 로젠블라트가 발표함
* 이진 분류(binary classification): 임의의 샘플 데이터를 True나 False로 구분하는 문제
```

* 퍼셉트론의 전체 구조를 훑어봅니다

  * 퍼셉트론은 직선 방정식을 사용하기 때문에 선형 회귀와 유사한 구조를 갖고 있지만 마지막 단계에서 계단 함수를 사용함

  * 계단 함수(step function): 샘플을 이진 분류하기 위해 사용하는 함수로 다시 가중치와 절편을 업데이트(학습)하는데 사용

  * 퍼셉트론의 전체 구조

    ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section01/github/image01.PNG?raw=true)

    * 뉴런 - 입력 신호들을 받아 z를 만듦

      * 선형 함수

        ![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap04/section01/github/image02.PNG?raw=true)

    * 계단 함수 - z가 0보다 크거나 같으면 1로, 0보다 작으면 -1로 분류함

      * y = 1 (z>=0), y = -1 (z<0)
      * 양성 클래스(positive class): 1, 음성 클래스(negative class): -1

  * 퍼셉트론은 선형 함수를 통과한 값 z를 계단 함수로 보내 1과 -1로 분류하는 알고리즘으로 결과를 통해 가중치와 절편을 업데이트

<br>

* 지금부터 여러 개의 특성을 사용하겠습니다



### 02. 아달린에 대해 알아봅니다

### 03. 로지스틱 회귀에 대해 알아봅니다

