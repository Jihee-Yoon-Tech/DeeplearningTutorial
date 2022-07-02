Introduction
============

# **딥러닝 데이터 표현과 연산**

<br>

- 데이터 표현을 위한 기본 구조로 텐서(Tensor)를 사용
- 텐서는 데이터를 담기 위한 컨테이너(container)로서 일반적으로 수치형 데이터를 저장

<br>

## 텐서 (Tensor)

- Rank: 축의 개수
- Shape: 형상(각 축에 따른 차원 개수)
- Type: 데이터 타입

> 텐서는 다차원 배열(Multi-dimensional Array)이다.

> 텐서는 벡터(Vector)와 행렬(Matrix)을 일반화한 것이며, 3차원 이상으로 확장할 수 있다.

> 텐서는 TensorFlow의 가장 주요한 객체이며, TensorFlow의 작업은 주로 텐서의 연산으로 이루어진다. TensorFlow는 텐서를 정의하고 연산을 수행하도록 하는 프레임워크 (Framework)이다.


## 텐서플로우 시작하기

```python

import numpy as np
import tensorflow as tf

```

<table>
<tr>
<th> Rank </th> <th> Example </th>
</tr>
<tr>
<td> 0D Tensor</td> <td> Scalar</td>
</tr>
<tr>
<td> 1D Tensor</td> <td> Vector</td>
</tr>
<tr>
<td> 2D Tensor</td> <td> Matrix</td>
</tr>
<tr>
<td> 3D Tensor</td> <td> 데이터가 연속된 시퀀스 데이터, 시간 축이 포함된 시계열 데이터, 주식 가격 데이터셋, 시간에 따른 질병 발병 데이터 등이 존재</td>
</tr>
<tr>
<td> 4D Tensor</td> <td> 컬러 이미지 데이터가 대표적인 사례(흑백 이미지 데이터는 3D Tensor로 가능)</td>
</tr>
<tr>
<td> 5D Tensor</td> <td> 비디오 데이터가 대표적인 사례</td>
</tr>
</table>

<details>
<summary> 텐서의 개념, 차원, 데이터 예시 </summary>

##

## 0D Tensor(i.e. Scalar)

- 하나의 숫자를 담고 있는 텐서(tensor)
- 축과 형상이 없음

```python
t0 = tf.constant(1)
print(t0)
print(tf.rank(t0))

## tf.Tensor(1, shape=(), dtype=int32)
## tf.Tensor(0, shape=(), dtype=int32)
```
<br>

## 1D Tensor(Vector)

- 값들을 저장한 리스트와 유사한 텐서
- 하나의 축이 존재

```python
t1 = tf.constant([1, 2, 3])
print(t1)
print(tf.rank(t1))

## tf.Tensor([1,2,3], shape=(3,), dtype=int32)
## tf.Tensor(1, shape=(), dtype=int32)
```

<br>

## 2D Tensor(Matrix)

- 행렬과 같은 모양으로 두개의 축이 존재
- 일반적인 수치, 통계 데이터셋이 해당
- 주로 샘플(samples)과 특성(features)을 가진 구조로 사용

```python
t2 = tf.constant([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(t2)
print(tf.rank(t2))

## tf.Tensor([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], shape=(3,3), dtype=int32)
## tf.Tensor(2, shape=(), dtype=int32)
```

<br>


## 3D Tensor

- 큐브(cube)와 같은 모양으로 세 개의 축이 존재
- 데이터가 연속된 시퀀스 데이터나 시간 축이 포함된 시계열 데이터에 해당
- 주식 가격 데이터셋, 시간에 따른 질병 발병 데이터 등이 존재
- 주로 샘플(samples), 타임스텝(timesteps), 특성(features)을 가진 구조로 사용

```python
t3 = tf.constant([[[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]],
                  [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]],
                  [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]])
print(t3)
print(tf.rank(t3))

## tf.Tensor([[[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]
                  
                  [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]
                  
                  [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]], shape=(3,3,3), dtype=int32)
## tf.Tensor(3, shape=(), dtype=int32)
```

<br>

## 4D Tensor

- 4개의 축
- 컬러 이미지 데이터가 대표적인 사례(흑백 이미지 데이터는 3D Tensor로 가능)
- 주로 샘플(samples), 높이(height), 너비(width), 컬러 채널(channel)을 가진 구조로 사용

<br>

## 5D Tensor

- 5개의 축
- 비디오 데이터가 대표적인 사례
- 주로 샘플(samples), 프레임(frames), 높이(height), 너비(width), 컬러 채널(channel)을 가진 구조로 사용

</details>

<br>

<details>
<summary> 텐서의 데이터 타입</summary>

##

## 텐서의 데이터 타입

<br>

- 텐서의 기본 타입
    - 정수형 텐서: int32
    - 실수형 텐서: float32
    - 문자열 텐서: string
- int32, float32, string 타입 외에도 float16, int8 타입 등이 존재
- 연산시 텐서의 타입 일치 필요
- 타입 변환에는 tf.cast() 사용

<br>

```python
i = tf.constant(2)
print(i)

## tf.Tensor(2, shape=(), dtype=int32)

f = tf.constant(2.)
print(f)

## tf.Tensor(2.0, shape=(), dtype=float32)

s = tf.constant('Suan')
print(s)

## tf.Tensor(b'Suan', shape=(), dtype=string)

f16 = tf.constant(2., dtype=tf.float16)
print(f16)

## tf.Tensor(2.0, shape=(), dtype=float16)

i8 = tf.constant(2, dtype=tf.int8)
print(i8)

## tf.Tensor(2.0, shape=(), dtype=int8)

```

</details>
<br>

<details>
<summary> 텐서의 연산</summary>


```python
print(tf.constant(2) * tf.constant(2))
print(tf.constant(2) / tf.constant(2))
print(tf.multiply(tf.constant(2), tf.constant(2)))
print(tf.divide(tf.constant(2), tf.constant(2)))


## tf.Tensor(4, shape=(), dtype=int32)
## tf.Tensor(1.0, shape=(), dtype=float64)
## tf.Tensor(4, shape=(), dtype=int32)
## tf.Tensor(1.0, shape=(), dtype=float64)
```

텐서플로우는 다른 타입의 텐서에 대해서 연산을 지원하지 않는다.

```python

print(tf.constant(2) + tf.constant(2.2))

## InvalidArgumentError: cannot compute AddV2 as ...

```

다른 타입의 경우 텐서를 같은 타입으로 cast 해준 뒤 연산을 수행해야 한다.

```python
print(tf.cast(tf.constant(2), tf.float32) + tf.constant(2.2))

## tf.Tensor(4.2, shape=(), dtype=float32)

```


</details>

<br>



# **딥러닝 구조 및 학습**

- 딥러닝 구조와 학습에 필요한 요소
    - 모델(네트워크)을 구성하는 레이어(Layer)
    - 입력 데이터와 그에 대한 목적(결과)
    - 학습 시에 사용할 피드백을 정의하는 손실함수(loss function)
    - 학습 진행 방식을 결정하는 옵티마이저(optimizer)

<br>

## 레이어(Layer)

- 신경망의 핵심 데이터 구조
- 하나 이상의 텐서를 입력받아, 하나 이상의 텐서를 출력하는 데이터 처리 모듈
- 상태가 없는 레이어도 있지만, 대부분 가중치(weight)라는 레이어 상태를 가짐
- 가중치는 확률적 경사 하강법에 의해 학습되는 하나 이상의 텐서

<br>

- Keras에서 사용되는 주요 레이어
    - Dense
    - Activation
    - Flatten
    - Input
    
```python
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
```


## Dense

- 완전 연결 계층(Fully-Connected Layer)
- 노드수(유닛수), 활성화 함수(activation) 등을 지정
- name을 통한 레이어간 구분 가능
- 가중치 초기화(kernel_initializer)
    - 신경망의 성능에 큰 영향을 주는 요소
    - 보통 가중치의 초기값으로 0에 가까운 무작위 값 사용
    - 특정 구조의 신경망을 동일한 학습 데이터로 학습시키더라도, 가중치의 초기값에 따라 학습된 신경망의 성능 차이가 날 수 있음.
    - 오차역전파 알고리즘은 기본적으로 경사하강법을 사용하기 때문에 최적해가 아닌 지역해에 빠질 가능성이 있음.
    - Keras에서는 기본적으로 Glorot uniform 가중치(Xavier 분포 초기화), zeros bias로 초기화
    - kernel_initializer 인자를 통해 다른 가중치 초기화 지정 가능
    - Keras에서 제공하는 가중치 초기화 종류: https://keras.io/api/layers/initializers/, https://keras.io/ko/initializers/

<br>

```python
Dense(10, activation='softmax')
Dense(10, activation='relu', name='Dense Layer')
Dense(10, kernel_initializer='he_normal', name='Dense Layer')
```

## Activation

- Dense Layer에서 미리 활성화 함수를 지정할 수도 있지만, 별도 레이어를 만들어줄 수 있음
- Keras에서 제공하는 활성화함수: https://keras.io/ko/activations/

<br>

<img src="https://miro.medium.com/max/1192/1*4ZEDRpFuCIpUjNgjDdT2Lg.png">


```python
dense = Dense(10, activation='relu', name='Dense Layer')
Activation(dense)
```

<br>

## Flatten

- 배치 크기(또는 데이터 크기)를 제외하고 데이터를 1차원으로 쭉 펼치는 작업
- 예시

```
(128, 3, 2, 2) -> (128, 12)
```

## Input

- 모델의 입력을 정의
- shape, dtype을 포함
- 하나의 모델은 여러 개의 입력을 가질 수 있음
- summary() 메소드를 통해서는 보이지 않음

```python
Input(shape=(28, 28), dtype=tf.float32)
```

```python
Input(shape=(8,), dtype=tf.int32)
```

<br>

## 모델(Model)

- 딥러닝 모델은 레이어로 만들어진 비순환 유향 그래프(Directed Acyclic Graph, DAG) 구조

<br>

## 모델 구성

- Sequential()
- 서브클래싱(subclassing)
- 함수형 api

<br>

## Sequential()

- 모델이 순차적인 구조로 진행할 때 사용
- 간단한 방법
    - Sequential 객체 생성 후 add()를 이용한 방법
    - Sequential 인자에 한 번에 추가 방법
- 다중 입력 및 출력이 존재하는 등의 복잡한 모델을 구성할 수 없음

<br>

```python
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model

model = Sequential()
model.add(Input(shape=(28, 28)))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

plot_model()
```
<br>

```python

model = Sequential([Input(shape=(28,28), name="Input"),
                    Dense(300, activation="relu", name="Dense1"),
                    Dense(100, activation="relu", name="Dense2"),
                    Dense(10, activation="softmax", name="Output")])
model.summary()
plot_model(model)
```
<br>

## 함수형 API

- 가장 권장되는 방법
- 모델을 복잡하고, 유연하게 구성 가능
- 다중 입출력을 다룰 수 있음

<br>

```python
inputs = Input(shape=(28, 28, 1))
x = Flatten(input_shape=(28, 28, 1))(inputs)
x = Dense(300, activation="relu")(x)
x = Dense(100, activation="relu")(x)
x = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
plot_model()
```

<br>

```python
from tensorflow.keras.layers import Concatenate
input_layer = Input(shape=(28, 28))
hidden1 = Dense(100, activation="relu")(input_layer)
hidden2 = Dense(30, activation="relu")(hidden1)
concat = Concatenate()([input_layer, hidden2])
output = Dense(1)(concat)

model = Model(inputs=[input_layer], outputs=[output])
model.summary()

plot_model()
```




<br>


<br>


#### References

- https://youtu.be/28QbrkRkHlo
- https://codetorial.net/tensorflow/basics_of_tensor.html

