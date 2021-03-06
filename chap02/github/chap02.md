# chap 02 최소한의 도구로 딥러닝을 시작합니다

2021.03.10

```
구글이 제공하는 구글 서비스 주피터 노트북인(jupyter notebook) 구글 코랩(colab)을 이용하여 실습 진행
```

<br>

### 02-1 구글 코랩을 소개합니다

* 코랩이란?

  * 구글에서 교육과 과학 연구를 목적으로 개발한 도구로 파이썬 코드를 실행하는 등 다양한 기능 사용 가능
  * 코랩은 웹 브라우저를 통해 제어하고 실제 파이썬 코드 실행은 구글 클라우드의 가상 서버에서 이루어지며 코랩에서 만든 파일(노트북)은 구글 드라이브에 저장됨

* 코랩에 접속해 기본 기능 익히기

  * [코랩 접속하기](https://colab.research.google.com/)

  * 코랩의 기본 기능 익히기

    ![image01](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap02/github/image01.PNG?raw=true)

* 코랩에서 노트북 관리하기

* 코랩 노트북에서 자주 사용하는 기능 알아보기

  1. 셀 삭제하기 - Ctrl + M D
  2. 셀 실행하기 - Ctrl + Enter
  3. 셀 실행하고 바로 다음 셀로 이동하기 - Shift + Enter
  4. 셀 실행하고 바로 아래에 새 셀 삽입하기 - Alt Enter
  5. 단축기 설정하기 - Ctrl + M H
  6. 명령 팔레트 사용하기 - [도구 > 명령 팔레트], Ctrl + Shift + P

<br>

### 02-2 딥러닝을 위한 도구들을 알아봅니다

* 넘파이를 소개합니다

  * 넘파이(Numpy): 파이썬의 핵심 과학 패키지 중 하나

    * 많은 머신러닝과 딥러닝 패키지가 넘파이를 기반으로 구현되었고 인터페이스를 계승하고 있음

    * 파이썬 리스트 복습하기

      * 2개의 숫자와 하나의 문자열로 이루어진 파이썬 리스트

        ```python
        my_list = [10, 'hello list', 20]
        print(my_list[1])	# 'hello list'
        ```

      * 파이썬의 2차원 배열

        ```python
        my_list_2 = [[10, 20, 30], [40, 50, 60]]
        print(my_list_2[1][1])	# 50
        ```

* 넘파이 준비하기

  * 파이썬 리스트로 만든 배열은 배열의 크기가 커질수록 성능이 떨어진다는 단점이 있는 반면 넘파이는 저수준 언어로 다차원 배열을 구현하여 크기가 커져도 높은 성능을 보장하므로 이 경우 넘파이를 사용함

    1. 코랩에서 넘파이 임포트하고 버전 확인하기

       ```python
       import numpy as np
       print(np.__version__)	# 1.19.5
       ```

* 넘파이로 배열 만들기

  * 넘파이는 파이썬의 리스트와 달리 숫자와 문자열을 함께 담을 수 없음

    1. array() 함수로 2차원 배열 만들기

       ```python
       my_arr = np.array([[10, 20, 30], [40, 50, 60]])
       print(my_arr)
       # [[10 20 30]
       #  [40 50 60]]
       ```

    2. type() 함수로 넘파이 배열인지 확인하기

       ```python
       type(my_arr)	# numpy.ndarray
       ```

    3. 넘파이 배열에서 요소 선택하기

       ```python
       my_arr[0][2]	# 30
       ```

    4. 넘파이 내장 함수 사용하기

       ```python
       np.sum(my_arr)	# 210
       ```


<br>

* 맷플롯립으로 그래프 그리기

  * 맷플롯립(Matplotlib): 파이썬 과학 생태계의 표준 그래프 패키지

    * 맷플롯립에서 제공하는 기능으로 대부분의 그래프를 그릴 수 있으며 코랩에 포함되어 바로 사용 가능

      ```python
      import matplotlib.pyplot as plt
      ```

    * 선 그래프와 산점도 (x축을 기준으로 y축의 변화 추이를 살펴보기 편리해 데이터 분석할 때 자주 사용)

      1. 선 그래프 그리기

         * 선 그래프를 그리려면 x축의 값과 y축의 값을 맷플롯립의 plot() 함수에 전달해야함

         ```python
         plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])	
         # x축, y축의 값을 파이썬 리스트로 전달합니다.
         plt.show()
         ```

         ![image02](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap02/github/image02.PNG?raw=true)

      2. 산점도 그리기

         * 산점도는 데이터의 x축, y축 값을 이용하여 점으로 그래프를 그린 것

         ```python
         plt.scatter([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])	
         plt.show()
         ```

         ![image03](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap02/github/image03.PNG?raw=true)

      3. 넘파이 배열로 산점도 그리기

         * 넘파이의 random.randn() 함수를 사용하여 표준 정규 분포를 따르는 난수의 산점도 그리기

         ```python
         x = np.random.randn(1000)	# 표준 정규 분포를 따르는 난수 1000개를 만듭니다.
         y = np.random.randn(1000)	# 표준 정규 분포를 따르는 난수 1000개를 만듭니다.
         plt.scatter(x, y)	
         plt.show()
         ```

         ![image04](https://github.com/hyunmin0317/DeepLearning_Study/blob/master/chap02/github/image04.PNG?raw=true)

  * 데이터를 분석하고 시각화하면 데이터에서 직관을 얻기 쉬워 딥러닝에서 데이터 시각화는 필수이다.
