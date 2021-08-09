#softmax에서 class 개수가 2개면 Logistic과 같음
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(1)
#//////////////////////////////
#기능테스트
#//////////////////////////////
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
y = torch.randint(5, (3,)) # 0, 2, 1
y_one_hot = torch.zeros_like(hypothesis) 
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
# 10000
# 00100
# 01000
### 수식을 그대로 표현
# 1. cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
### log_softmax는 log와 softmax를 함께 실행
# 2. cost = (y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()
### nll_loss는 one hot vector없이 바로
# 3. cost = F.nll_loss(F.log_softmax(z, dim=1), y)
### cross entropy는 한번에 모두
# 4.
cost = F.cross_entropy(z, y)
print(cost)
#//////////////////////////////
#기능테스트 끝
#//////////////////////////////

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
# https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/3
# 왜 Long을 쓰는가

model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    prediction = model(x_train)
    cost = F.cross_entropy(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        classification = torch.LongTensor([list(prediction[i]).index(max(list(prediction[i]))) for i in range(len(prediction))])
        correct_prediction = classification == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100
        ))