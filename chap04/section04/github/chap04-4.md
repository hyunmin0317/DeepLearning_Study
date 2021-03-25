# chap 04-4 분류용 데이터 세트를 준비합니다

2021.03.25

```python
* 분류 문제를 위해 데이터 세트로 사이킷런에 포함된 '위스콘신 유방암 데이터 세트(Wisconsin breast cancer dataset)' 사용
```

<br>

### 01. 유방암 데이터 세트를 소개합니다

* 유방암 데이터 세트: 유방암 세포의 특징 10개에 대하여 평균, 표준 오차, 최대 이상치가 기록된 데이터 세트

  * 유방암 데이터 세트를 통해 유방암 데이터 샘플이 악성 종양(True)인지 정상 종양(False)인지를 구분하는 이진 분류 문제 해결

  * 용어 정리

    |      |         의학         |       이진 분류       |
    | :--: | :------------------: | :-------------------: |
    | 좋음 | 양성 종양(정상 종양) |       음성 샘플       |
    | 나쁨 |      악성 종양       | 양성 샘픔 (해결 과제) |

<br>

### 02. 유방암 데이터 세트 준비하기

1. load_breast_cancer() 함수 호출하기

   * 사이킷런의 datasets 모듈 아래에 있는 load_breast_cancer() 함수를 사용하여 Bunch 클래스의 유방암 데이터 세트를 불러옴

     ```python
     from sklearn.datasets import load_breast_cancer
     cancer = load_breast_cancer()
     ```

2. 입력 데이터 확인하기

   * cancer의 data와 target 살펴보기

     ```python
     print(cancer.data.shape, cancer.target.shape)	# 입력 데이터인 data의 크기 (569, 30) (569,)
     cancer.data[:3]	# 처음 3개의 샘플 출력
     ```

   * 특성 데이터는 양수인 실수 범위의 값이며 30개의 특성으로 구성되어 있어 산점도로 나타내기 어려우므로 박스 플롯(box plot)을 이용하여 각 특성의 사분위(quartile) 값을 나타내 보겠습니다.

3. 박스 플롯으로 특성의 사분위 관찰하기

   * 박스 플롯: 1사분위와 3사분위 값으로 상자를 그린 다음 그 안에 2사분위(중간값) 값을 표시하고 1사분위와 3사분위 사이 거리의 1.5배만큼 위아래 거리에서 각각 가장 큰 값과 가장 작은 값까지 수염을 그림

   * 유방암 데이터 세트를 이용하여 박스 플롯을 그리기

     ```python
     plt.boxplot(cancer.data)
     plt.xlabel('feature')
     plt.ylabel('value')
     plt.show()
     ```

     ![image01]

     <br>

4. 눈에 띄는 특성 살펴보기

   * 다른 특성과 차이가 나는 특성들 확인 (박스 플롯의 특성 중 다른 특성보다 값의 분포가 훨씬 큰 4, 14, 24번째 특성 확인)

     ```python
     cancer.feature_names[[3,13,23]]
     # array(['mean area', 'area error', 'worst area'], dtype='<U23') -> 넓이와 관련된 특성
     ```

5. 타깃 데이터 확인하기

   ```python
   np.unique(cancer.target, return_counts=True)	# (array([0, 1]), array([212, 357]))
   ```

   * 이진 분류 문제이므로 cancer.target 배열 안에는 0과 1만 들어있음
   * unique() 함수는 고유한 값을 찾아 반환하는 함수를 return_counts 매개변수가 True이므로 고유한 값의 갯수를 반환

6. 훈련 데이터 세트 저장하기

   * 예제 데이터 세트를 x, y 변수에 저장

     ```python
     x = cancer.data
     y = cancer.target
     ```

   * 훈련 데이터 세트 준비를 마쳤으니 로지스틱 회귀를 이용하여 모델을 만들어보겠습니다.

   