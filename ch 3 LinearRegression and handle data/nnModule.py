import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1) # input_dim, output_dim

    # 입력 받은 값에 대해 연산을 수행하기 위한 함수
    def forward(self, x):
        return self.linear(x)

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

    # 입력 받은 값에 대해 연산을 수행하기 위한 함수
    def forward(self, x): 
        return self.linear(x)

# 단일 변수
''' 
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
model = LinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 2000
for epoch in range(epochs+1):

        # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    cost.backward() # backward 연산
    optimizer.step() # W와 b를 업데이트

    # 200번마다 로그 출력
    if epoch % 200 == 0:
      print(
          f'Epoch {epoch:4d}/{epochs} Cost: {cost.item():.6f}'
      )

new_var = torch.FloatTensor([4.0])
pred_y = model(new_var)
print("훈련 후 입력이 4일 때의 예측값 :", pred_y) 
'''
#다중 변수

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
model = MultivariateLinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 200 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

new_var = torch.FloatTensor([73,80,75])
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
print(list(model.parameters()))