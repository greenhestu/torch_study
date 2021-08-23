print("\n----numpy----\n")
#/\/\/\/\ numpy로 직접구현 /\/\/\/\
import numpy as np

timesteps = 10 # 시점의 수. NLP에서는 주로 문장의 길이.
input_size = 4 # 입력의 차원. NLP에서는 주로 단어 벡터의 차원.
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량.

inputs = np.random.random((timesteps, input_size)) # 입력(2D 텐서)

hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태 0(벡터)로 초기화
# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.

Wx = np.random.random((hidden_size, input_size))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wh = np.random.random((hidden_size, hidden_size)) # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
b = np.random.random((hidden_size,)) # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).

total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs: # 각 시점에 따라서 입력값이 입력됨.
  output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh,hidden_state_t) + b) # Wx * Xt + Wh * Ht-1 + b(bias)
  total_hidden_states.append(list(output_t)) # 각 시점의 은닉 상태의 값을 계속해서 축적
  print(np.shape(total_hidden_states)) # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
  hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis = 0) 
# 출력 시 값을 깔끔하게 해준다.

print(total_hidden_states) # (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.
#/\/\/\/\ numpy로 직접구현 /\/\/\/\

print("\n----torch_RNN----\n")

#/\/\/\/\ torch를 이용해 구현 /\/\/\/\
import torch
import torch.nn as nn

input_size = 5 #입력 크기
hidden_size = 8 #은닉상태 크기

# (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(input_size, hidden_size, batch_first=True)
outputs, _status = cell(inputs) # 1st return : all hidden state, 2nd return : last hidden state
print(outputs.shape, _status.shape)

print("\n층이 2개 이상이면")
cell = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
outputs, _status = cell(inputs) # 1st : last layer's all hidden state, 2nd : last hidden state
print(outputs.shape, _status.shape)

print("\n양방향인 경우 (time step당 셀이 2개)")
cell = nn.RNN(input_size = 5, hidden_size = 8, num_layers = 2, batch_first=True, bidirectional = True)
outputs, _status = cell(inputs) 
# (배치 크기, 시퀀스 길이, 은닉 상태의 크기 x 2), (층의 개수 x 2, 배치 크기, 은닉 상태의 크기)
print(outputs.shape, _status.shape)
print(outputs)
print(_status)
#/\/\/\/\ torch를 이용해 구현 /\/\/\/\

print("\n----torch_LSTM----\n")
cell = nn.LSTM(input_size, hidden_size, batch_fisrt=True)  