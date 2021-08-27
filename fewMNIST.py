# smile lab에 있을 때,
# 학습데이터의 수를 줄여가며 정확도의 변화를 확인하기 위한 파일 
# class당 500개 -> 98.59%
# class당 50개 -> 79.25%
# class당 100개 -> 89.22%
# class당 150개 -> 93.64%
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 랜덤 시드 고정
RANDOM_SEED = 777
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(RANDOM_SEED)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

class CustomDataset(Dataset): 
    x_data = []
    y_data = []
    labels = [0,0,0,0,0,0,0,0,0,0]
    def __init__(self, datas):
        for data in datas:
            if(self.labels[data[1]] < 150): # class당 data 수 제한하기
                self.x_data.append(data[0])
                self.y_data.append(data[1])
                self.labels[data[1]] += 1
        print(self.labels)
    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)
    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        y = self.y_data[idx]#self.y_data[idx])
        x = torch.FloatTensor(self.x_data[idx])
        return (x, y)



class myModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7*7*64, 10, bias = True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out =self.layer1(x)
        out =self.layer2(out)
        out =out.view(out.size(0), -1)
        out =self.fc(out)
        return out 


## 1 * 28 * 28
## conv1
## 1 * 32 * 28 * 28
## pool 
## 1 * 32 * 14 * 14
## conv2
## 1 * 64 * 14 *14
## pool
## 1 * 64 * 7 * 7
## 1 * 3136
## fc
## 10 (0부터 9까지)

mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=False)

mnist_train = CustomDataset(mnist_train)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=False)

data_loader = DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)

model = myModule().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0
    for mini_batch, label in data_loader:
        mini_batch = mini_batch.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        hypothesis = model(mini_batch)
        cost = criterion(hypothesis, label)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())