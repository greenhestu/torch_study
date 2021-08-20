import torch

# 원-핫 벡터 생성
dog = torch.FloatTensor([1, 0, 0, 0, 0])
cat = torch.FloatTensor([0, 1, 0, 0, 0])
computer = torch.FloatTensor([0, 0, 1, 0, 0])
netbook = torch.FloatTensor([0, 0, 0, 1, 0])
book = torch.FloatTensor([0, 0, 0, 0, 1])

print(torch.cosine_similarity(dog, cat, dim=0))
print(torch.cosine_similarity(cat, computer, dim=0))
print(torch.cosine_similarity(computer, netbook, dim=0))
print(torch.cosine_similarity(netbook, book, dim=0))

# 원-핫 벡터로는 단어간 연관성을 찾을 수 없기에 
# CBOW와 Skip Gram을 사용한다.
# CBOW는 여러 단어를 받아 사이에 들어갈 단어를 추측한다
#   W행렬로 작은 차원(각 값은 실수)으로 만든 후에 소프트맥스
# Skip Gram은 한 단어를 받아 그 단어와 같이 사용되는 단어를 추측한다
#   일반적으로 CBOW보다 성능이 좋다고 함

# 단어 수가 많아지면 embedding이 복잡해지기때문에
# Negative Sampling이 사용된다
# 단어 집합을 줄이고 Binary문제로 바꿔줌