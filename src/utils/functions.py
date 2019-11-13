import sys
import numpy as np
from .alphabet import Alphabet
NULLKEY = "-null-"
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def read_instance_with_gaz(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet, number_normalized, max_sent_length, word_sense_map, char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    ent_cnt = 0
    ent_multi_cnt = 0
    UNK_id = gaz_alphabet.get_index(gaz_alphabet.UNKNOWN)
    sid = '-1'
    chr_ids = []

    total_num = 0
    valid_num = 0
    actual_max_len = -1

    for idx in range(len(in_lines)):
        line = in_lines[idx]        
        if len(line) > 2:
            if len(line)>4 and line[:4]=='sid:':
                sid = line[4:].strip()
                continue
            elif len(words) >= max_sent_length:
                continue
            pairs = line.strip().split()
            word = pairs[0]
            if len(pairs)==3:
                chr_ids.append(int(pairs[1]))

            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1].split(',')
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]

            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append([label_alphabet.get_index(lbl) for lbl in label])
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:

                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        else:
            total_num += 1
            actual_max_len = max(actual_max_len,len(words))
            if ((max_sent_length < 0) or (len(words) <= max_sent_length)) and (len(words)>0):
                gazs = []
                gaz_Ids = []
                w_length = len(words)

                for idx in range(w_length):
                    
                    matched_list = gaz.enumerateMatchList(words[idx:])  #以widx开始直到结束，连续组成的串是否组成词
                    matched_Id = []
                    matched_length = []
                    for entity in matched_list:
                        if gaz.space:
                            entity = entity.split(gaz.space)
                        entlen = len(entity)
                        entity = ''.join(entity) 
                        ent_ind = gaz_alphabet.get_index(entity)
                        ent_cnt += 1 
                        if ent_ind == UNK_id: #没找到，代表多义
                            if word_sense_map and entity in word_sense_map:
                                ent_multi_cnt += 1
                                for cur_ent in word_sense_map[entity]:
                                    cur_ind = gaz_alphabet.get_index(cur_ent)
                                    matched_Id.append(cur_ind)
                                    matched_length.append(entlen)
                        else:
                            matched_Id.append(ent_ind)
                            matched_length.append(entlen)
                    gazs.append(matched_list)

                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])

                instence_texts.append([words, biwords, chars, gazs, labels, [sid,chr_ids]])
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids])
                valid_num += 1
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
            gazs = []
            gaz_Ids = []
            sid = '-1'
            chr_ids = []
    print('Given max sent length:',max_sent_length,' Actual:',actual_max_len)
    print('Total ent:',ent_cnt,' Ent with multi-sense:',ent_multi_cnt,' Ratio:',str(100.0*ent_multi_cnt/(ent_cnt+1e-6))+'%')
    print('Total instance:',total_num,' Valid instance:',valid_num)
    return instence_texts, instence_Ids


def read_instance_with_gaz_in_sentence(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r').readlines()
    instence_texts = []
    instence_Ids = []
    for idx in xrange(len(in_lines)):
        pair = in_lines[idx].strip().split()

        orig_words = list(pair[0])
        
        if (max_sent_length > 0) and (len(orig_words) > max_sent_length):
            continue
        biwords = []
        biword_Ids = []
        if number_normalized:
            words = []
            for word in orig_words:
                word = normalize_word(word)
                words.append(word)
        else:
            words = orig_words
        word_num = len(words)
        for idy in range(word_num):
            if idy < word_num - 1:
                biword = words[idy]+words[idy+1]
            else:
                biword = words[idy]+NULLKEY
            biwords.append(biword)
            biword_Ids.append(biword_alphabet.get_index(biword))
        word_Ids = [word_alphabet.get_index(word) for word in words]
        label = pair[-1]
        label_Id =  label_alphabet.get_index(label)
        gazs = []
        gaz_Ids = []
        word_num = len(words)
        chars = [[word] for word in words]
        char_Ids = [[char_alphabet.get_index(word)] for word in words]

        for idx in range(word_num):
            matched_list = gaz.enumerateMatchList(words[idx:])
            matched_length = [len(a) for a in matched_list]
            gazs.append(matched_list)
            matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]
            if matched_Id:
                gaz_Ids.append([matched_Id, matched_length])
            else:
                gaz_Ids.append([])
        instence_texts.append([words, biwords, chars, gazs, label])
        instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Id])
    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


       
def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if len(tokens) <= 3:
                continue
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
