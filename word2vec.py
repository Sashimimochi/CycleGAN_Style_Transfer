import gensim, logging
import os
import json

class MySentences(object):
  def __init__(self, filename, dfile):
    self.filename = filename
    self.dfile = dfile
 
  def __iter__(self):
    word = []
    for l in open(self.dfile):
      word.append(l.strip())

    for line in open(self.filename):
      sent = line.split(' +++$+++ ')[1].split()
      sent = [w if w in word else '__UNK__' for w in sent]
      sent = ['__BOS__'] + sent
      for _ in range(12-len(sent)):
        sent = sent + ['__EOS__']
      sent[-1] = '__EOS__'
      yield sent

sentences = MySentences('data/source_train', 'data/word')
model = gensim.models.Word2Vec(sentences,size=200,window=5,min_count=5,workers=7, iter=10)
model.save('data/word_vec')

i = 0
word_id_dict = dict()
for word in model.wv.vocab:
  word_id_dict[word] = i
  i += 1
fp = open('./data/dict','w')
json.dump(word_id_dict,fp, ensure_ascii=False)

