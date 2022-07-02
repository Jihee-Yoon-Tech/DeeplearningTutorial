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
<td> 0D Tensor</td> <td> Scalar</td>
</tr>
<tr>
<td> 1D Tensor</td> <td> Vector</td>
</tr>
<tr>
<td> 2D Tensor</td> <td> Matrix</td>
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



</details>

<br>

#### References

- https://youtu.be/28QbrkRkHlo
- https://codetorial.net/tensorflow/basics_of_tensor.html

