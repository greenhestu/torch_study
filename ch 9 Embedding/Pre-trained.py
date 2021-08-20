import gensim

#/\/\/\/\ 영어 /\/\/\/\ 
'''
# 구글의 사전 훈련된 Word2Vec 모델을 로드합니다.
pre_trained_path_en  = '??????/GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_path_en, binary=True) 

print(model.vectors.shape)
print(model.similarity('this', 'is'))
print(model.similarity('post', 'book'))
'''

#/\/\/\/\ 한국어 /\/\/\/\
# 한국어 모델 로드
pre_trained_path_kr = '??????/ko.bin'
model = gensim.models.Word2Vec.load(pre_trained_path_kr)

result = model.wv.most_similar("강아지")
print(result)