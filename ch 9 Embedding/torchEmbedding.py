# pytorch에서도 nn.Embedding()을 통해 임베딩 벡터를 사용할 수 있다.
# 단어를 one-hot encoding하지 않고 정수로 변환한다.
# 정수를 index로 삼아 look-up table에서 찾는다.
# look-up table을 학습을 통해 최적화 시킨다.

train_data = 'you need to know how to code'
word_set = set(train_data.split()) # 중복을 제거한 단어들의 집합인 단어 집합 생성.
vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}  # 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab['<unk>'] = 0
vocab['<pad>'] = 1

import torch.nn as nn

embedding_layer = nn.Embedding(
    num_embeddings = len(vocab), 
    embedding_dim = 3,
    padding_idx = 1
    )
# num_embeddings : 임베딩을 할 단어들의 개수. 다시 말해 단어 집합의 크기입니다.
# embedding_dim : 임베딩 할 벡터의 차원입니다. 사용자가 정해주는 하이퍼파라미터입니다.
# padding_idx : 선택적으로 사용하는 인자입니다. 패딩을 위한 토큰의 인덱스를 알려줍니다.

#이제 이 값으로 학습해야 함 (학습 전 상태)
print(embedding_layer.weight)


#/\/\/\/\/\/\ 훈련 데이터를 가져와서 사용하자 /\/\/\/\/\/\
from torchtext.legacy import datasets
from torchtext.legacy import data
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

# 1. 저장된 파일을 이용해 모델 불러오기
from gensim.models import keyedvectors
word2vec_saved = 'eng_w2v'
word2vec_model = keyedvectors.load_word2vec_format(word2vec_saved)\

import torch
import torch.nn as nn # 위에도 있는데 여기서 쓰여서 리마인더겸 다시 씀
from torchtext.vocab import Vectors
vectors = Vectors(name=word2vec_saved) # 사전 훈련된 Word2Vec 모델을 vectors에 저장

TEXT.build_vocab(trainset, vectors=vectors, max_size=10000, min_freq=10) # Word2Vec 모델을 임베딩 벡터값으로 initialize
# 기존에 있던 단어는 임베딩 벡터가 유지되고, 없던 단어는 전부 0으로 초기화됨 

embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
# freeze가 true면 가중치 업데이트를 안 함
print(f"{word2vec_saved}에서 불러온 모델")
print(embedding_layer(torch.LongTensor([10]))) # 단어 this의 임베딩 벡터값
#이제 embedding_layer를 data의 text, label로 학습시킬 수 있음

# 2. torchtext에서 불러오기
from torchtext.vocab import GloVe
# glove.6B.300d를 download
TEXT.build_vocab(trainset, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(trainset)
print(TEXT.vocab.stoi)
embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
print("torchtext에서 불러온 모델")
print(embedding_layer(torch.LongTensor([10]))) # 단어 this의 임베딩 벡터값