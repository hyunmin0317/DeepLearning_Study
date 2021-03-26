# chap 04-7 사이킷런으로 로지스틱 회귀를 수행합니다

2021.03.26

```markdown
* SGDClassifier 클래스: 로지스틱 회귀 문제 외에도 여러 가지 문제에 경사 하강법을 적용할 수 있는 클래스
```

<br>

### 01. 사이킷런으로 경사 하강법 적용하기

1. 로지스틱 손실 함수 지정하기

   ```python
   from sklearn.linear_model import SGDClassifier
   sgd = SGDClassifier(loss='log', max_iter=100, tol=1e-3, random_state=42)
   ```

   * 매개변수 설명
     * loss = 'log': 로지스틱 회귀를 적용하기 위해 loss 매개변수에 손실 함수로 log를 지정
     * max_iter=100: 반복 횟수를 100으로 지정
     * random_state=42: 반복 실행했을 때 결과를 동일하게 재현하기 위해 random_state를 통해 난수 초깃값을 42로 설정
     * tol=1e-3: 반복할 때마다 로지스틱 손실 함수의 값이 tol에 지정한 값만큼 감소되지 않으면 반복을 중단함

2. 사이킷런으로 훈련하고 평가하기

   ```python
   sgd.fit(x_train, y_train)	# 훈련
   sgd.score(x_test, y_test)	# 0.833333333333334 (정확도 계산)
   ```

   * 사이킷런의 SGDClassifier 클래스의 메서드는 지금까지 우리가 직접 구현한 클래스의 메서드와 동일함

3. 사이킷런으로 예측하기 (데이터 세트에 대한 예측)

   ```python
   sgd.predict(x_test[0:10]) # array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
   ```

   * 사이킷런은 입력 데이터로 2차원 배열만 받아들임