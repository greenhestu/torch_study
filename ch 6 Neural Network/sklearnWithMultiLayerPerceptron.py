import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
print('전체 샘플의 수 : {}'.format(len(digits.images)))

import torch
import torch.nn as nn
from torch import optim
torch.manual_seed(777)
model = nn.Sequential(
    nn.Linear(64, 32), # input_layer = 64, hidden_layer1 = 32
    nn.ReLU(),
    nn.Linear(32, 16), # hidden_layer2 = 32, hidden_layer3 = 16
    nn.ReLU(),
    nn.Linear(16, 10) # hidden_layer3 = 16, output_layer = 10
)

X = digits.data # 이미지. 즉, 특성 행렬
Y = digits.target # 각 이미지에 대한 레이블
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss() # 이 비용 함수는 소프트맥스 함수를 포함하고 있음.
optimizer = optim.Adam(model.parameters()) # Adaptive하게 learning rate를 조절함.
losses = []

for epoch in range(100):
  optimizer.zero_grad()
  y_pred = model(X) # forward 연산
  loss = loss_fn(y_pred, Y)
  loss.backward()
  optimizer.step()

  if epoch % 10 == 0:
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 100, loss.item()
        ))

  losses.append(loss.item())

  images_and_labels = list(zip(digits.data, digits.images, digits.target))
  rand = int(torch.randint(low = 0,high = len(images_and_labels), size = (1,1)))
for index, (data, image, label) in enumerate(images_and_labels[rand:rand+5]): # 5개의 샘플만 출력
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i' % label)
    print(model(torch.tensor(data, dtype=torch.float32)).argmax())
plt.show()