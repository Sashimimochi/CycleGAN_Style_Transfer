import os
import numpy as np
import json
import re
import gensim
import random
from flags import FLAGS

class utils():
    def __init__(self,args):
        self.data_dir = FLAGS.data_dir
        self.data_train = os.path.join(self.data_dir, 'source_train.txt')
        self.data_test = os.path.join(self.data_dir, 'source_test.txt')
        self.num_batch = args.dis_it + args.gen_it + 1
        self.sent_length = args.sequence_length
        self.batch_size = args.batch_size
        
        self.set_dictionary(os.path.join(self.data_dir, 'dict.json'))
        self.set_word2vec_model(os.path.join(self.data_dir, 'word_vec.model'))


    def set_dictionary(self, dict_file):
        if os.path.exists(dict_file):
            fp = open(dict_file,'r')
            self.word_id_dict = json.load(fp)
            print('word number:',len(self.word_id_dict))

            self.BOS_id = self.word_id_dict['__BOS__']
            self.EOS_id = self.word_id_dict['__EOS__']
            self.UNK_id = self.word_id_dict['__UNK__']

            self.id_word_dict = [[]]*len(self.word_id_dict)
            for word in self.word_id_dict:
                self.id_word_dict[self.word_id_dict[word]] = word
        else:
            print('where is dictionary file QQ?')


    def set_word2vec_model(self,name):
        word_array = []
        self.word2vec_model = gensim.models.Word2Vec.load(name)
        for i in range(len(self.id_word_dict)):
            word = self.id_word_dict[i]
            try:
                word_array.append(self.word2vec_model[word])
            except KeyError as e:
                print('KeyError at word of: ', word)
                print('[ERROR]', e)
                raise KeyError(e)

        self.word_array = np.array(word_array)


    def sent2id(self,sent):
        vec = np.ones((self.sent_length),dtype=np.int32) * self.EOS_id

        i = 0
        for word in sent.split():
            if word in self.word_id_dict:
                vec[i] = self.word_id_dict[word]
            else:
                vec[i] = self.UNK_id
            i += 1
            if i>=self.sent_length:
                break
        return vec

    def vec2sent(self,vecs):
        sent = []
        for vec in vecs:
            possible_words = self.word2vec_model.most_similar([vec],topn=10)
            word = possible_words[0][0]
            if word != '__EOS__':
              sent.append(word)
            else:
              break
        if sent == []:
          sent = ['.']
        return ''.join(sent)


    def id2sent(self,indices):
        sent = []
        for index in indices:
            sent.append(self.id_word_dict[index])
        return ' '.join(sent)

    """
    def data_generator(self,class_id):
        while(1):
            with open(self.data_train) as fp:
                for line in fp:
                    s = line.strip().split(' +++$+++ ')
                    if int(s[0])==class_id and random.randint(0,10) >= 2:
                        yield self.sent2id(s[1].strip())

    def X_data_generator(self):
        return self.data_generator(0)


    def Y_data_generator(self):
        return self.data_generator(1)
    """

    def gan_data_generator(self):
        while(1):
            one_X_batch = []
            one_Y_batch = []
            with open(self.data_train) as fp:
                for line in fp:
                    s = line.strip().split(' +++$+++ ')
                    if int(s[0])==0 and random.randint(0,10) >= 3 and len(one_X_batch) < self.batch_size*self.num_batch:
                        one_X_batch.append(self.sent2id(s[1].strip()))
                    elif int(s[0])==1 and random.randint(0,10) >= 3 and len(one_Y_batch) < self.batch_size*self.num_batch:
                        one_Y_batch.append(self.sent2id(s[1].strip()))

                    if len(one_X_batch) == self.batch_size*self.num_batch and len(one_Y_batch) == self.batch_size*self.num_batch:
                        one_X_batch = np.array(one_X_batch).reshape(self.num_batch,self.batch_size,-1)
                        one_Y_batch = np.array(one_Y_batch).reshape(self.num_batch,self.batch_size,-1)
                        yield one_X_batch,one_Y_batch
                        one_X_batch = []
                        one_Y_batch = []

    def pretrain_generator_data_generator(self):

        while(1):
            one_X_batch = []
            one_Y_batch = []
            with open(self.data_train) as fp:
                for line in fp:
                    s = line.strip().split('+++$+++')
                    if int(s[0])==0 and random.randint(0,10) >= 3 and len(one_X_batch) < self.batch_size:
                        one_X_batch.append(self.sent2id(s[1].strip()))
                    elif int(s[0])==1 and random.randint(0,10) >= 3 and len(one_Y_batch) < self.batch_size:
                        one_Y_batch.append(self.sent2id(s[1].strip()))

                    if len(one_X_batch) == self.batch_size and len(one_Y_batch) == self.batch_size:
                        one_X_batch = np.array(one_X_batch)
                        one_Y_batch = np.array(one_Y_batch)
                        yield one_X_batch,one_Y_batch
                        one_X_batch = []
                        one_Y_batch = []

    def test_data_generator(self):
        one_batch = np.ones([self.batch_size,self.sent_length])
        sentx = []
        senty = []
        batch_count = 0
        for line in open(os.path.join(FLAGS.data_dir, 'source_test.txt')):
            x, y = line.strip().split(' +++$+++ ')
            one_batch[batch_count] = self.sent2id(y)
            sentx.append(''.join(x.split()))
            senty.append(y)
            batch_count += 1
            if batch_count == self.batch_size:
                yield one_batch, sentx, senty
                batch_count = 0
                one_batch = np.zeros([self.batch_size,self.sent_length])
                sentx = []
                senty = []

        if batch_count >= 1:
            yield one_batch, sentx, senty

