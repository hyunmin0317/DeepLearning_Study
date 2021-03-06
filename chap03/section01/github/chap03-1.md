# chap 03-1 선형 회귀에 대해 알아보고 데이터를 준비합니다

2021.03.11

<br>

### 01. 1차 함수로 이해하는 선형 회귀

```markdown
* 선형 회귀(Linear Regression): 머신러닝 알고리즘 중 가장 간단하면서도 딥러닝의 기초가 되는 알고리즘
```

* 선형 회귀는 y=ax+b 와 같은 1차 함수로 (기울기는 a이고 절편은 b) 표현할 수 있음
* 선형 회귀는 x와 y의 값이 주어졌을 때 기울기와 절편을 찾아 모델을 만든 다음 새 입력에 대해 어떤 값을 예상해 문제를 해결하는 과정

<br>

### 02. 선형 회귀를 통한 현실적인 문제 해결

```markdown
* 당뇨병 환자의 1년 후 병의 진전된 정도를 예측하는 모델 만들기 예제
```

* 문제 해결을 위해 당뇨병 환자의 데이터 준비하기

  * 머신러닝, 딥러닝 패키지에는(사이킷런,케라스) 인공지능 학습을 위한 데이터 세트가 준비되어 있고 예제에서는 사이킷런의 당뇨병 환자 데이터 세트 사용

  1. load_diabetes() 함수로 당뇨병 데이터 준비하기

     ```python
     from sklearn.datasets import load_diabetes
     diabetes = load_diabetes()	
     # diabetes에 당뇨병 데이터 저장, 자료형은 파이썬 딕셔너리와 유사한 Bunch 클래스
     ```

     

  2. 입력과 타깃 데이터의 크기 확인하기

     ```python
     print(diabetes.data.shape, diabetes.target.shape)	# (442, 10) (442,)
     ```

     * 필요한 입력과 타깃 데이터가 diabetes의 data 속성과 target 속성에 넘파이 배열로 저장되어 있음
     * data는 442개의 행과 10개의 열로 구성된 2차원 배열이고 target은 442개 요소를 가진 1차원 배열
     * 행은 샘플(sample)로 환자에 대한 특성으로 이루어진 데이터 1세트를 의미하고 열은 샘플의 특성(feature)으로 데이터의 여러 특징을 의미함
     * 입력 데이터의 특성은 속성, 독립 변수(independent variable), 설명 변수(explanatory variable) 등으로 부름

  3. 입력 데이터 자세히 보기

     ```python
     diabetes.data[0:3]	# 슬라이싱을 사용해 입력 데이터 앞부분의 샘플 3개만 출력
     ```

  4. 타깃 데이터 자세히 보기

     ```python
     diabetes.target[:3] # [0:3] == [:3], 슬라이싱을 사용해 타깃 데이터 앞부분의 타깃 3개만 출력
     ```

     * 이 예제에서 타깃 데이터는 10개의 요소로 구성된 샘플 1개에 대응됨

<br>

### 03. 당뇨병 환자 데이터 시각화하기

1. 맷플롯립의 scatter() 함수로 산점도 그리기

   * 이 예제에서 당뇨병 데이터 세트에는 10개의 특성이 있지만 3차원 이상의 그래프는 그릴 수 없으므로 1개의 특성만 사용

   * 세 번째 특성과 타깃 데이터로 그린 산점도

     ```python
     import matplotlib.pyplot as plt
     plt.scatter(diabetes.data[:, 2], diabetes.target)
     plt.xlabel('x')	# diabetes.data의 세 번째 특성
     plt.ylabel('y')	# diabetes.target
     plt.show()
     ```

     ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap03/section01/github/image01.PNG?raw=true)

     * 입력 데이터와 타깃 데이터 사이에 정비례 관계가 있음을 확인

2. 훈련 데이터 준비하기

   * 매번 입력 데이터를 입력하여 속성을 참고하는 방법은 번거로우므로 입력 데이터를 미리 분리하여 x에 타깃 데이터는 변수 y에 저장하여 사용

     ```python
     x = diabetes.data[:, 2]
     y = diabetes.target
     ```

* 이번 장에서는 회귀 알고리즘 중 선형 회귀 알고리즘의 개념을 알아보며 실제 알고리즘을 만들어보기 위한 데이터 세트를 준비했고 다음 장에서는 이 데이터를 통해 모델을 훈련하기 위한 핵심 최적화 알고리즘인 경사 하강법(gradient descent)에 대해 배움