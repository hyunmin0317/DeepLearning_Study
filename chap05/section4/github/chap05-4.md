# chap 05-4 교차 검증을 알아보고 사이킷런으로 수행해 봅니다

2021.03.26

```markdown
* 교차 검증(cross validation): 훈련 세트의 샘플 개수가 줄어들어 모델을 훈련시킬 데이터가 부족해지는 경우 사용
```

<br>

### 01. 교차 검증의 원리를 알아봅니다

![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap05/section4/github/image01.PNG?raw=true)

* 교차 검증은 훈련 세트를 작은 덩어리로 나누어 진행하며 훈련 세트를 나눈 작은 덩어리를 '폴드'라고 부름
* 교차 검증 과정
  * 훈련 세트를 k개의 폴드(fold)로 나눕니다.
  * 첫 번째 폴드를 검증 세트로 사용하고 나머지 폴드(k-1개)를 훈련 세트로 사용합니다.
  * 모델을 훈련한 다음에 검증 세트로 평가합니다.
  * 차례대로 다음 폴드를 검증 세트로 사용하여 반복합니다.
  * k개의 검증 세트로 k번 성능을 평가한 후 계산된 성능의 평균을 내어 최종 성능을 계산합니다.
* 기존의 훈련 방법보다 더 많은 데이터로 훈련할 수 있음

<br>

### 02. k-폴드 교차 검증을 구현합니다

1. 훈련 세트 사용하기

   ```python
   validation_scores = []
   ```

   * k-폴드 교차 검증은 검증 세트가 훈련 세트에 포함되어 전체 데이터 세트를 다시 훈련 세트와 테스트 세트로 1번만 나눈 x_train_all과 y_train_all을 훈련과 검증에 사용하며 각 폴드의 검증 점수를 저장하기 위해 validation_scores 리스트를 정의

2. k-폴드 교차 검증 구현하기

   ```python
   k = 10
   bins = len(x_train_all) // k
   
   for i in range(k):
       start = i*bins
       end = (i+1)*bins
       val_fold = x_train_all[start:end]
       val_target = y_train_all[start:end]
       
       train_index = list(range(0, start))+list(range(end, len(x_train_all)))
       train_fold = x_train_all[train_index]
       train_target = y_train_all[train_index]
       
       train_mean = np.mean(train_fold, axis=0)
       train_std = np.std(train_fold, axis=0)
       train_fold_scaled = (train_fold - train_mean) / train_std
       val_fold_scaled = (val_fold - train_mean) / train_std
       
       lyr = SingleLayer(l2=0.01)
       lyr.fit(train_fold_scaled, train_target, epochs=50)
       score = lyr.score(val_fold_scaled, val_target)
       validation_scores.append(score)
   
   print(np.mean(validation_scores))	# 0.9711111111111113
   ```

   * 훈련 데이터의 표준화 전처리를 폴드를 나눈 후에 수행하며 반복문을 진행하여 검증 폴드로 측정한 성능 점수를 리스트에 저장
   * 성능 점수들의 평균을 내기 위해 np.mean() 함수를 사용함
   * k-폴드 교차 검증을 통해 얻은 성능 점수는 이전의 방법으로 검증하여 얻은 점수보다 안정적이고 조금 더 신뢰할 수 있음

<br>

### 03. 사이킷런으로 교차 검증을 합니다

1. cross_validate() 함수로 교차 검증 점수 계산하기

   ```python
   from sklearn.model_selection import cross_validate
   sgd = SGDClassifier(loss='log', penalty='l2', alpha=0.001, random_state=42)
   scores = cross_validate(sgd, x_train_all, y_train_all, cv=10)
   print(np.mean(scores['test_score']))	# 0.850096618357488
   ```

   * 매개변수 값으로 교차 검증을 하고 싶은 모델의 객체와 훈련 데이터, 타깃 데이터를 전달하고 cv 매개변수에 교차 검증을 수행할 폴드 수를 지정하여 cross_validate() 함수 사용
   * 표준화 전처리를 수행하지 않았기 때문에 교차 검증의 평균 점수가 낮음

<br>

### 04. 전처리 단계 포함해 교차 검증을 수행합니다

* Pipeline 클래스 사용해 교차 검증 수행하기

  ```python
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  pipe = make_pipeline(StandardScaler(), sgd)
  scores = cross_validate(pipe, x_train_all, y_train_all, cv=10, return_train_score=True)
  print(np.mean(scores['test_score']))	# 0.9694202898550724
  ```

  * 표준화 전처리 단계와 SGDClassifier 클래스 객체를 Pipeline 클래스로 감싸 cross_validate() 함수에 전달함
  * cross_validate() 함수는 훈련 세트를 훈련 폴드와 검증 폴드로 나누며 전처리 단계는 Pipeline 클래스 객체에서 이루어짐
  * cross_validate() 함수에 return_train_score 매개변수를 True로 설정하면 훈련 폴드의 점수도 얻을 수 있음

* 훈련 폴드의 점수

  ```python
  print(np.mean(scores['train_score']))	# 0.9875478561631581
  ```

  