# ML Lightning Template

Deep Learning 모델을 PyTorch Lightning을 활용해서 쉽게 만들 수 있는 템플릿 코드를 만든다.

## Objective

모델 학습, 저장 등에 대한 신경은 쓰지 않고, 모델 그 자체(nn.Module)만 잘 만들면, 바로 학습하고 활용할 수 있는 수준으로 패키지화한다.

## 예상되는 문제점

크게 3가지 형태로 다양한 form을 갖게 된다.
1. (모듈 관점) Transformer, CNN, LSTM등의 모듈 근간이 다른 경우
2. (거시적인 태스크 관점) NLP, Vision에 따라서 추상화 계층이 다르다.
3. (세부적인 태스크 관점) Objective Detection, Face Recognition에 따른 추상화 계층이 다른 경우

따라서, 각 변형에 알맞는 추가적인 추상화가 필요할 수 있다.
해당 템플릿은 추가적인 추상화 이전의 추상계층을 만드는 것을 목적으로 한다.

## 아키텍쳐

대부분의 인터페이스는 PyTorch Lightning의 인터페이스를 따른다.
PyTorch Lightning을 한단계 더 추상화하여

## 목표

### 첫번째 목표
MNIST 모델을 Backbone으로 하는 템플릿 코드

### End Goals
Face Recognition 모델

## 아키텍쳐
아직은 막연하게 DI, IoC개념을 적극적으로 활용하는 것을 고려하고있다.
