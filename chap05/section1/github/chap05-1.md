# chap 05-1 검증 세트를 나누고 전처리 과정을 배웁니다

2021.03.26

<br>

### 01. 테스트 세트로 모델을 튜닝합니다

* 로지스틱 회귀로 모델 훈련하고 평가하기

  * 데이터 세트를 읽어 들여 훈련 세트와 테스트 세트로 나누기

    ```python
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target
    x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
    ```

  * SGDClassifier 클래스를 이용하여 로지스틱 회귀 모델을 훈련

    ```python
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(loss='log', random_state=42)
    sgd.fit(x_train_all, y_train_all)
    sgd.score(x_test, y_test)	# 0.8333333333333334
    ```

    * fit() 메서드로 모델을 훈련하고 score() 메서드로 성능을 평가
    * 하이퍼파라미터(hyperparameter): 사용자가 직접 선택해야 하는 값으로 알아서 학습되지 않는 loss와 같은 매개변수를 말함

* 서포트 벡터 머신으로 모델 훈련하고 평가하기

  * SGDClassifier 클래스의 loss 매개변수를 log에서 hinge로 바꿈

    ```python
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(loss='hinge', random_state=42)
    sgd.fit(x_train_all, y_train_all)
    sgd.score(x_test, y_test)	# 0.9385964912280702
    ```

    * 선형 서포트 벡터 머신(SVM) 문제를 푸는 모델이 만들어짐
    * SVM(Support Vector Machine): 훈련 데이터의 클래스를 구분하는 경계선을 찾는 작업

* SGDClassifier 클래스의 매개변수들을 바꾸며 '모델을 튜닝'하면 테스트 성능을 높일 수 있지만 실전에서는 좋은 성능을 내지 못함  

<br>

### 02. 테스트 세트로 모델을 튜닝하면 실전에서 좋은 성능을 기대하기 어렵습니다

* 테스트 세트로 모델을 튜닝하면 '테스트 세트에 대해서만 좋은 성능을 보여주는 모델'이 만들어지고 실전에서 같은 성능을 기대하기 어려우며 이 현상을 '테스트 세트의 정보가 모델에 새어 나갔다' 라고 함
* 테스트 세트로 모델을 튜닝하면 모델의 일반화 성능(generalization performance)이 왜곡됨

<br>

### 03. 검증 세트를 준비합니다 

* 전체 데이터 세트를 훈련, 검증 , 테스트 세트로 나누고 모델을 훈련해야 함

  * 검증 세트(validation set): 모델을 튜닝하는 용도의 세트로 훈련 세트를 조금 떼어 만듦

* SGDClassifier 클래스로 만든 모델을 훈련

  1. 데이터 세트 준비하기 (위스콘신 유방암 데이터 사용)

     ```python
     from sklearn.datasets import load_breast_cancer
     from sklearn.model_selection import train_test_split
     cancer = load_breast_cancer()
     x = cancer.data
     y = cancer.target
     x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
     ```

  2. 검증 세트 분할하기

     ```python
     x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)
     print(len(x_train), len(x_val))	# 364 91
     ```

     * 전체 데이터 세트를 8:2로 나누어 훈련 세트와 테스트 세트를 만들고 훈련 세트를 8:2로 나누어 훈련 세트와 검증 세트를 만듦

  3. 검증 세트 사용해 모델 평가하기

     ```python
     sgd = SGDClassifier(loss='log', random_state=42)
     sgd.fit(x_train, y_train)
     sgd.score(x_val, y_val)	# 0.6923076923076923
     ```

     * 훈련 세트의 크기가 줄어들어 성능이 감소했고 위의 데이터 세트는 샘플 개수가 적은 편으로 검증 세트의 비율이나 매개변수의 값을 조금만 조절해도 성능 평가 점수가 크게 변함
     * 데이터 양이 적은 경우 검증 세트를 나누지 않는 교차 검증이라는 방법을 사용하지만 대부분 훈련, 검증, 테스트 세트로 나눔

<br>

### 04. 데이터 전처리와 특성의 스케일을 알아봅니다

```python
데이터 전처리(data preprocessing): 데이터를 적절히 가공하는 과정
    * 실전에서 수집된 데이터의 형태가 균일하지 않거나 누락된 값이 있을 때 데이터를 적절히 가공하는 데이터 전처리 과정이 필요함
```

* 특성의 스케일은 알고리즘에 영향을 줍니다

  * 이 책에서는 제대로 가공되지 않은 데이터를 다루지 않지만 특성의 스케일(scale)이 다른 경우 또한 데이터 전처리 과정이 필요함

  * 특성의 스케일(값의 범위)이 다른 경우

    |        | 당도 | 무게 |
    | :----: | :--: | :--: |
    | 사과 1 |  4   | 540  |
    | 사과 2 |  8   | 700  |
    | 사과 3 |  2   | 480  |

    * 사과의 당도 범위는 1~10이고 무게의 범위는 500~1000으로 두 특성의 스케일 차이가 큼
    * 어떤 알고리즘은(신경망 알고리즘 등) 스케일에 민감하여 모델의 성능에 영향을 줌

<br>

### 05. 스케일을 조정하지 않고 모델을 훈련해 볼까요?

1. 훈련 데이터 준비하고 스케일 비교하기
2. 가중치를 기록할 변수와 학습률 파라미터 추가하기
3. 가중치 기록하고 업데이트 양 조절하기
4. 최종으로 결정된 가중치

<br>

### 06. 스케일을 조정해 모델을 훈련합니다

1. 넘파이로 표준화 구현하기
2. 모델 훈련하기
3. 모델 성능 평가하기
4. 표준화 전처리를 적용

<br>

### 07. 스케일을 조정한 다음에 실수하기 쉬운 함정을 알아봅니다

1. 원본 훈련 세트와 검증 세트로 산점도 그리기
2. 전처리한 훈련 세트와 검증 세트로 산점도 그리기
3. 올바르게 검증 세트 전처리하기
4. 모델 평가하기