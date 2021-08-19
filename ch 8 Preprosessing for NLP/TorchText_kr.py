import urllib.request
import pandas as pd
from torchtext.legacy import data # torchtext.data 임포트
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator
from konlpy.tag import Mecab

# 처음 한 번만
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_df = pd.read_table('ratings_train.txt')
test_df = pd.read_table('ratings_test.txt')
print('훈련 데이터 샘플의 개수 : {}'.format(len(train_df)))
print('테스트 데이터 샘플의 개수 : {}'.format(len(test_df)))

tokenizer = Mecab()
batch_size = 5

# 필드 정의
ID = data.Field(sequential = False,
                use_vocab = False # 실제 사용은 하지 않을 예정
                ) 

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer.morphs, # 토크나이저로는 Mecab 사용.
                  lower=True,
                  batch_first=True,
                  fix_length=20
                  )
# batch_first가 false라면
# data1, data2, data3이 아니라
# (data1[0],data2[0],data3[0]), (data1[1],data2[1],..), ..가 된다.

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True
                   )

train_data, test_data = TabularDataset.splits(
        path='.', 
        train='ratings_train.txt', 
        test='ratings_test.txt', 
        format='tsv',
        fields=[('id', ID), ('text', TEXT), ('label', LABEL)], 
        skip_header=True
        )

TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)