import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

input_str = 'apple' # problem
label_str = 'pple!' # solution
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)
print ('문자 집합의 크기 : {}'.format(vocab_size))

input_size = vocab_size # problem size
hidden_size = 5
output_size = 5 # label size
learning_rate = 0.1

char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 문자에 고유한 정수 인덱스 부여
print(char_to_index)

index_to_char={}
# 나중에 복호화하기 위한 set
for key, value in char_to_index.items():
    index_to_char[value] = key
print(index_to_char)

x_data = [char_to_index[c] for c in input_str]
y_data = [char_to_index[c] for c in label_str]
print(x_data) # a, p, p, l, e
print(y_data) # p, p, l, e, !

# 배치 차원 추가, nn.RNN이 3차원을 입력받아서
# 텐서 연산인 unsqueeze(0)를 통해 해결할 수도 있음.
x_data = [x_data]
y_data = [y_data]

x_one_hot = [np.eye(vocab_size)[x] for x in x_data]
print(x_one_hot) # one_hot encoding

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True) # RNN 셀 구현
        self.fc = torch.nn.Linear(hidden_size, output_size, bias=True) # 출력층 구현

    def forward(self, x): # 구현한 RNN 셀과 출력층을 연결
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

net = Net(input_size, hidden_size, output_size)
outputs = net(X)
print(outputs.shape) # 3차원
print(outputs.view(-1, input_size).shape) # 2차원 텐서로 변환
print(Y.shape)
print(Y.view(-1).shape)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

for i in range(100):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs.view(-1, input_size), Y.view(-1)) # view를 하는 이유는 Batch 차원 제거를 위해
    loss.backward() # 기울기 계산
    optimizer.step() # 아까 optimizer 선언 시 넣어둔 파라미터 업데이트

    # 아래 세 줄은 모델이 실제 어떻게 예측했는지를 확인하기 위한 코드.
    result = outputs.data.numpy().argmax(axis=2) # 최종 예측값인 각 time-step 별 5차원 벡터에 대해서 가장 높은 값의 인덱스를 선택
    result_str = ''.join([index_to_char[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)