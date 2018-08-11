import numpy as np
from root_module.parameters_module.set_dir import Directory
from root_module.parameters_module.set_params import GlobalParams
import pickle
import os
import pandas as pd

import numpy as np

class Dictionary():
    
    def __init__(self, mode='TR'):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """
        self.mode = mode
        self.word_dict = None
        self.rel_dir = Directory(mode)
        # gloveDict = rel_dir.glove_path
        # pickle.load(open(self.rel_dir.glove_present_training_word_vocab, 'rb'))     
        if os.path.exists(self.rel_dir.word_vocab_dict):
            self.word_dict = pickle.load(open(self.rel_dir.word_vocab_dict, 'rb'))
            print('Picking the dictionary from: ', self.rel_dir.word_vocab_dict)
        else:
            self.__process()
        # Not planning to use pretrained embeddings
        # wordEmb = self.rel_dir.word_embedding
        # self.glove_present_word_csv = np.float32(np.genfromtxt(wordEmb, delimiter=' '))
        # self.label_dict = pickle.load(open(self.rel_dir.label_map_dict, 'rb'))
        

    def __process(self):
        print(self.rel_dir.raw_base_file)
        
        # f = open(self.rel_dir.raw_base_file, encoding='ascii', mode='r')
        print('Making the dictionary from: ', self.rel_dir.raw_base_file)
        f = open(self.rel_dir.raw_base_file, mode='r')
        # data = pd.read_csv(self.rel_dir.raw_base_file, sep='\t')
        # data = f.readlines().strip()
        # samples = data.iloc[:, 0]
        # samples = data
        vocab = dict()
        # vocab[''] = (0, len(samples))
        vocab['UNK'] = (1, 0)
        numWords = 2
        
        # for sample in samples:
        for sample in f:
            words = sample.split()
            # We want to get unique words
            for word in list(set(words)):
                dictValue = vocab.get(word)
                if dictValue is None:
                    vocab[word] = (numWords, 1)
                    numWords += 1
                else:
                    code, count = dictValue
                    vocab[word] = (code, count+1)
        
        # Using the empty to keep the size of the dictionary
        print('Size of the dictionary: ', len(vocab))
        vocab[''] = (0, len(vocab))
        self.word_dict = vocab
        
        # Should this update the value of the vocabulary size? 
        global_params = GlobalParams()
        global_params.vocab_size = len(vocab)
        f.close()
                                               
        # Saving to the dictionary file
        with open(self.rel_dir.word_vocab_dict, 'wb') as vf:
            pickle.dump(self.word_dict, vf)
            