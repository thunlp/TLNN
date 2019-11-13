import sys
import numpy as np
from .alphabet import Alphabet
from .functions import *
import pickle
from .gazetteer import Gazetteer
import json

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"

class Data:
    def __init__(self): 
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.word_sense_map = None
        self.sense_word_map = None
        self.norm_gaz_emb = False
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        # self.word_alphabet.add(START)
        # self.word_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(START)
        # self.char_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(PADDING)
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM" 

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.use_bigram = True
        self.word_emb_dim = 50
        self.biword_emb_dim = 50
        self.char_emb_dim = 30
        self.gaz_emb_dim = 50
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 160
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = False
        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0

        
    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Use          bigram: %s"%(self.use_bigram))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Biword alphabet size: %s"%(self.biword_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Gaz   alphabet size: %s"%(self.gaz_alphabet.size()))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Biword embedding size: %s"%(self.biword_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Gaz embedding size: %s"%(self.gaz_emb_dim))
        print("     Norm     word   emb: %s"%(self.norm_word_emb))
        print("     Norm     biword emb: %s"%(self.norm_biword_emb))
        print("     Norm     gaz    emb: %s"%(self.norm_gaz_emb))
        print("     Norm   gaz  dropout: %s"%(self.gaz_dropout))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("     Hyperpara  iteration: %s"%(self.HP_iteration))
        print("     Hyperpara  batch size: %s"%(self.HP_batch_size))
        print("     Hyperpara          lr: %s"%(self.HP_lr))
        print("     Hyperpara    lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s"%(self.HP_clip))
        print("     Hyperpara    momentum: %s"%(self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s"%(self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s"%(self.HP_bilstm))
        print("     Hyperpara         GPU: %s"%(self.HP_gpu))
        print("     Hyperpara     use_gaz: %s"%(self.HP_use_gaz))
        print("     Hyperpara fix gaz emb: %s"%(self.HP_fix_gaz_emb))
        print("     Hyperpara    use_char: %s"%(self.HP_use_char))
        if self.HP_use_char:
            print("             Char_features: %s"%(self.char_features))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file,'r',encoding='utf-8').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                labels = pairs[-1].split(',')
                for label in labels:
                    self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s"%(old_size, self.label_alphabet_size))

    def build_word_sense_map(self, input = None):
        if input:
            fr = open(input, 'r', encoding = 'utf-8')
            self.word_sense_map = dict()
            self.sense_word_map = dict()
            for line in fr:
                if len(line) < 5:
                    continue
                line = line.strip().split(' ')
                self.word_sense_map[line[0]] = set(line[1:])
                for sense in line[1:]:
                    self.sense_word_map[sense] = line[0]

    def build_alphabet(self, input_file):
        in_lines = open(input_file,'r',encoding='utf-8').readlines()
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                if 'sid:' in line:
                    continue
                pairs = line.strip().split()
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                labels = pairs[-1].split(',')
                for label in labels:
                    self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                if idx < len(in_lines) - 1 and len(in_lines[idx+1]) > 2:
                    biword = word + in_lines[idx+1].strip().split()[0]
                else:
                    biword = word + NULLKEY
                self.biword_alphabet.add(biword)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"


    def build_gaz_file(self, gaz_file):
        ## build gaz file,initial read gaz embedding file
        if gaz_file:
            visit = set()
            fins = open(gaz_file, 'r', encoding='utf-8').readlines()
            for fin in fins:
                fin = fin.strip().split()[0]
                #fin = fin.strip().split()[0].decode('utf-8')
                if fin:
                    if self.sense_word_map:
                        if fin in self.sense_word_map:
                            fin = self.sense_word_map[fin]
                        if fin in visit:
                            continue
                        visit.add(fin)
                    self.gaz.insert(fin, "one_source")
            print("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            print("Gaz file is None, load nothing")


    def build_gaz_alphabet(self, input_file):
        in_lines = open(input_file,'r',encoding='utf-8').readlines()
        word_list = []
        entity_cnt = 0
        multi_sense_cnt = 0
        for line in in_lines:
            if len(line) > 3:
                if 'sid:' in line:
                    continue
                word = line.split()[0]
                #word = line.split()[0].decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                word_list.append(word)
            else:
                w_length = len(word_list)
                for idx in range(w_length):
                    matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                    for entity in matched_entity:
                        if self.gaz.space:
                            entity = ''.join(entity.split(self.gaz.space))
                        entity_cnt += 1
                        if self.word_sense_map and entity in self.word_sense_map:
                            if len(self.word_sense_map[entity]) > 1:
                                multi_sense_cnt += 1
                            for sense in self.word_sense_map[entity]:
                                self.gaz_alphabet.add(sense)
                        else:
                            self.gaz_alphabet.add(entity)
                        # print entity, self.gaz.searchId(entity),self.gaz.searchType(entity)
                        #self.gaz_alphabet.add(entity)
                word_list = []
        print("input file:", input_file, " Total words in gaz:",entity_cnt," Words with multi-sense:",multi_sense_cnt)
        print("gaz alphabet size:", self.gaz_alphabet.size())

    def build_gold_dict(self,name,gold_path):
        if gold_path:
            fr = open(gold_path,'r',encoding='utf-8')
            gold = dict()
            if '.golden.dat' in gold_path:
                for line in fr:
                    line = line.strip().split("\t")
                    key = line[0],int(line[1]),int(line[2])
                    if not key in gold:
                        gold[key] = []
                    gold[key].append(line[4].upper())
            else:
                tmp = json.load(fr) #sid -> [[offset,length,word,label]]
                
                for sid,vals in tmp.items():
                    for offset,length,word,label in vals:
                        key = (sid,int(offset),int(length))
                        if key not in gold:
                            gold[key] = []
                        gold[key].append(label.upper())
        else:
            gold = None
            
        if name == "train":
            self.train_gold = gold
        elif name == "dev":
            self.dev_gold = gold
        elif name == "test":
            self.test_gold = gold  
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close() 
        self.gaz_alphabet.close()  


    def build_word_pretrain_emb(self, emb_path):
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet, self.biword_emb_dim, self.norm_biword_emb)

    def build_gaz_pretrain_emb(self, emb_path):
        print("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet,  self.gaz_emb_dim, self.norm_gaz_emb)


    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def generate_instance_with_gaz(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.gaz_alphabet,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH, self.word_sense_map)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(input_file, self.gaz,self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.gaz_alphabet,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH, self.word_sense_map)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.gaz_alphabet,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH, self.word_sense_map)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,self.biword_alphabet, self.char_alphabet, self.gaz_alphabet,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH, self.word_sense_map)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file,'w',encoding='utf-8')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sid,predict_result = predict_results[idx]
            sent_length = len(predict_result)
            if isinstance(sid,list):
                sid,_ = sid
            fout.write('sid:'+sid+'\n')
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy] + " " + ','.join(content_list[idx][4][idy]) + " " + predict_result[idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, output_file))





