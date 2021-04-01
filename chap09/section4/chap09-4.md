# chap 09-4 LSTM 순환 신경망을 만들고 텍스트를 분류합니다

2021.04.01

<br>

### 01. LSTM 셀의 구조를 알아봅니다

* LSTM 셀의 구조

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section4/image01.PNG?raw=true" style="zoom: 67%;" />

* LSTM 셀 계산 수행 과정

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section4/image02.PNG?raw=true" style="zoom: 50%;" />

* 전체 공식

  <img src="https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap09/section4/image03.PNG?raw=true" style="zoom: 67%;" />

<br>

### 02. 텐서플로로 LSTM 순환 신경망 만들기

1. LSTM 순환 신경망 만들기
2. 모델 훈련하기
3. 손실 그래프와 정확도 그래프 그리기
4. 검증 세트 정확도 평가하기