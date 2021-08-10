# L1 : W 합이 cost에 추가
# L2 : W 제곱합이 cost에 추가

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
epochs = 15
batch_size = 64

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=False)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=False)

# dataset loader
loader_train = DataLoader(dataset=mnist_train,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)
loader_test = DataLoader(dataset=mnist_test,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True)

model = nn.Sequential().to(device)
model.add_module('fc1', nn.Linear(28*28*1, 100, device=device))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100, device=device))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10, device=device))

# 오차함수 선택
loss_fn = nn.CrossEntropyLoss().to(device)
# 가중치를 학습하기 위한 최적화 기법 선택
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()  # 신경망을 학습 모드로 전환

    # 데이터로더에서 미니배치를 하나씩 꺼내 학습을 수행
    for data, targets in loader_train:
        data = data.view(-1, 28*28).to(device)
        targets = targets.to(device)
        optimizer.zero_grad()  # 경사를 0으로 초기화
        #print(next(model.parameters()).is_cuda);
        outputs = model(data)  # 데이터를 입력하고 출력을 계산
        loss = loss_fn(outputs, targets)  # 출력과 훈련 데이터 정답 간의 오차를 계산
        loss.backward()  # 오차를 역전파 계산
        optimizer.step()  # 역전파 계산한 값으로 가중치를 수정

    print("epoch{}：완료\n".format(epoch))

def test():
    model.eval()  # 신경망을 추론 모드로 전환
    correct = 0

    # 데이터로더에서 미니배치를 하나씩 꺼내 추론을 수행
    with torch.no_grad():  # 추론 과정에는 미분이 필요없음
        for data, targets in loader_test:
            data = data.view(-1, 28*28).to(device)
            targets = targets.to(device)
            outputs = model(data)  # 데이터를 입력하고 출력을 계산

            # 추론 계산
            _, predicted = torch.max(outputs.data, 1)  # 확률이 가장 높은 레이블이 무엇인지 계산
            correct += predicted.eq(targets.data.view_as(predicted)).sum()  # 정답과 일치한 경우 정답 카운트를 증가

    # 정확도 출력
    data_num = len(loader_test.dataset)  # 데이터 총 건수
    print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct,
                                                   data_num, 100. * correct / data_num))

for epoch in range(3):
    train(epoch)

test()

index = 2021
X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
Y_test = mnist_test.test_labels.to(device)
model.eval()  # 신경망을 추론 모드로 전환
data = X_test[index]
output = model(data)  # 데이터를 입력하고 출력을 계산
_, predicted = torch.max(output.data, 0)  # 확률이 가장 높은 레이블이 무엇인지 계산

print("예측 결과 : {}".format(predicted))

X_test_show = (X_test[index]).to('cpu').numpy()
plt.imshow(X_test_show.reshape(28, 28), cmap='gray')
plt.show()
print("이 이미지 데이터의 정답 레이블은 {:.0f}입니다".format(Y_test[index]))
