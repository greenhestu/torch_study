import urllib.request
import pandas as pd
from torchtext.legacy import data
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator
#처음 한 번만 실행
#urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
print(df.head())
print('전체 샘플의 개수 : {}'.format(len(df)))

#/\/\/\ train, test 분리 /\/\/\
train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

# 필드 정의
'''
sequential : 시퀀스 데이터 여부. (True가 기본값)
use_vocab : 단어 집합을 만들 것인지 여부. (True가 기본값)
tokenize : 어떤 토큰화 함수를 사용할 것인지 지정. (string.split이 기본값)
lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
is_target : 레이블 데이터 여부. (False가 기본값)
fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행된다.
'''
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

train_data, test_data = TabularDataset.splits(
        path='.', train='train_data.csv', test='test_data.csv', 
        format='csv', skip_header=True,
        fields=[('text', TEXT), ('label', LABEL)]
        )

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))
print(vars(train_data[0]))

'''
min_freq : 단어 집합에 추가 시 단어의 최소 등장 빈도 조건을 추가.
max_size : 단어 집합의 최대 크기를 지정.
'''
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))
print(TEXT.vocab.stoi)

batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)
print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))
print('테스트 데이터의 미니 배치 수 : {}'.format(len(test_loader)))
batch = next(iter(train_loader)) # 첫번째 미니배치
#unk 0은 단어사전에 없는 단어(드물게 나타나는 단어)
#pad 1은 그냥 padding