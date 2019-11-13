# -*- coding: utf-8 -*-

import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.metric import *
from models.bilstmcrf import BiLSTM_CRF as SeqModel
from utils.data import Data
import os
from config import *
from utils.data_builder import *

# Set the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def evaluate(data, model, name, use_scorer=False, ver='new'):

    if name == "train":
        instances = data.train_Ids
        texts = data.train_texts
        gold_dict = data.train_gold
    elif name == "dev":
        instances = data.dev_Ids
        texts = data.dev_texts
        gold_dict = data.dev_gold
    elif name == 'test':
        instances = data.test_Ids
        texts = data.test_texts
        gold_dict = data.test_gold
    elif name == 'raw':
        instances = data.raw_Ids
        texts = data.raw_texts
    else:
        print("Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []

    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end >train_num:
            end =  train_num
        instance = instances[start:end]
        text = texts[start:end]
        if not instance:
            continue
        gaz_list,batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(gaz_list,batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)


        if ver=='new':
            for idx in range(len(pred_label)):
                pred_label[idx] = [text[idx][5],pred_label[idx]]
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    if ver=='new':
        acc_span, p_span, r_span, f_span, acc_type, p_type, r_type, f_type = get_ner_fmeasure(gold_dict, gold_results, pred_results, data.tagScheme, use_scorer)
    else:
        acc_span, p_span, r_span, f_span, acc_type, p_type, r_type, f_type = get_ner_fmeasure_old(gold_results, pred_results, data.tagScheme)
    return speed, acc_type, p_type, r_type, f_type, acc_span, p_span, r_span, f_span, pred_results  


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]

    biwords = [sent[1] for sent in input_batch_list]

    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))

    max_seq_len = int(word_seq_lengths.max())    
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label = [lbl[0] for lbl in label]# 选第一个
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label) 
        mask[idx, :seqlen] = torch.Tensor([1 for i in range(seqlen)])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    

    max_word_len = max(list(map(max, length_list)))

    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    
    gaz_list = [ gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def train(data, save_model_dir, dataset, seg=True ):
    print("Start training...")
    data.show_data_summary()
    save_data_name = save_model_dir +".dset"
    save_data_setting(data, save_data_name)
    model = SeqModel(data)

    print('Finised building the model..')
    loss_function = nn.NLLLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)

    best_dev = -1 # TC
    best_dev2 = -1 # TI
    data.HP_iteration = 70

    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)

        model.train()
        model.zero_grad()
        batch_size = 1 
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size 
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            gaz_list,  batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)

            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)

            right, whole = predict_check(tag_seq, batch_label, mask)

            right_token += right
            whole_token += whole
            sample_loss += loss.data[0]
            total_loss += loss.data[0]
            batch_loss += loss

            if end%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                sys.stdout.flush()
                sample_loss = 0
            if end%data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))       
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))

        speed, acc, p, r, f, acc2, p2, r2, f2, _ = evaluate(data, model, "test", use_scorer= dataset.lower()=='kbp')
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if seg:
            current_score = f
            current_score2 = f2
            print("Dev: time: %.2fs, speed: %.2fst/s; Full tags acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f))
            print("Default tags acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(acc2, p2, r2, f2))
        else:
            current_score = acc
            current_score2 = acc2
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))

        if current_score > best_dev:
            if seg:
                print(">>>> Exceed previous best f score (Full tags):", best_dev)
            else:
                print(">>>> Exceed previous best acc score:", best_dev)   
            if current_score2 <= best_dev2:    
                model_name = save_model_dir + '.' + str(idx) + '_fl_' + '{:.4g}_{:.4g}'.format(current_score2,current_score) +  ".model"
                torch.save(model.state_dict(), model_name)
            best_dev = current_score 
        if current_score2 > best_dev2:
            if seg:
                print(">>>> Exceed previous best f score (Default tags):", best_dev2)
            else:
                print(">>>> Exceed previous best acc score:", best_dev2)
            model_name = save_model_dir + '.' + str(idx) + '_df_' + '{:.4g}_{:.4g}'.format(current_score2,current_score) + ".model"
            torch.save(model.state_dict(), model_name)
            best_dev2 = current_score2
        gc.collect() 


def load_model_decode(model_dir, data, name, gpu, dataset, seg=True):
    data.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = SeqModel(data)

    model.load_state_dict(torch.load(model_dir))

    
    print("Decode %s data ..."%(name))
    start_time = time.time()
    speed, acc, p, r, f, acc2, p2, r2, f2, pred_results = evaluate(data, model, name, use_scorer= dataset.lower()=='kbp')
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print("%s: time:%.2fs, speed:%.2fst/s; Full tags acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
        if acc2 >= 0:
            print("Span tags acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(acc2, p2, r2, f2))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results


if __name__ == '__main__':
    
    dataset = os.path.join(public, dataset)
    train_file = os.path.join(dataset, TRAIN)
    dev_file = os.path.join(dataset, dev)
    test_file = os.path.join(dataset, test)
    raw_file = raw

    train_gold = os.path.join(dataset, train_gold) if train_gold is not None else None
    dev_gold = os.path.join(dataset, dev_gold) if dev_gold is not None else None
    test_gold = os.path.join(dataset, test_gold) if test_gold is not None else None
    
    model_dir = loadmodel
    dset_dir = savedset
    output_file = output
    word_sense_map_file = os.path.join(public, word_sense_map) if word_sense_map else None

    if seg.lower() == "true":
        seg = True 
    else:
        seg = False
    status = status.lower()

    save_model_dir = savemodel
    gpu = torch.cuda.is_available()
    maxlen = int(maxlen)


    char_emb = os.path.join(public, pretrain_char_emb)
    bichar_emb = None if bigram is None else os.path.join(public, bigram)

    if not word_sense_map_file is None:
        gaz_file = os.path.join(public, pretrain_sense_emb) 

    else:
        gaz_file = os.path.join(public, pretrain_word_emb) 

    print("CuDNN:", torch.backends.cudnn.enabled)
    # gpu = False
    print("GPU available:", gpu)
    print("Status:", status)
    print("Seg: ", seg)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Raw file:", raw_file)
    print("Char emb:", char_emb)
    print("Bichar emb:", bichar_emb)
    print("Gaz file:",gaz_file)
    if status == 'train':
        print("Model saved to:", save_model_dir)
    sys.stdout.flush()
    
    if status == 'train':
        data = Data()
        data.HP_gpu = gpu
        data.HP_lr = float(lr)
        data.HP_use_char = False
        data.HP_batch_size = 1
        data.use_bigram = False if bichar_emb is None else True 
        data.gaz_dropout = 0.5
        data.norm_gaz_emb = False
        data.HP_fix_gaz_emb = False
        data.MAX_SENTENCE_LENGTH = maxlen
        data_initialization(data, gaz_file, train_file, dev_file, test_file, word_sense_map_file)
        data_build_gold(data,train_gold,dev_gold,test_gold)
        data.generate_instance_with_gaz(train_file,'train')
        data.generate_instance_with_gaz(dev_file,'dev')
        data.generate_instance_with_gaz(test_file,'test')
        data.build_word_pretrain_emb(char_emb)
        data.build_biword_pretrain_emb(bichar_emb)
        data.build_gaz_pretrain_emb(gaz_file)
        train(data, save_model_dir, dataset, seg)
    elif status == 'test':      
        data = load_data_setting(dset_dir)
        data.build_word_sense_map(word_sense_map_file)
        data.MAX_SENTENCE_LENGTH = maxlen
        data_build_gold(data,train_gold,dev_gold,test_gold)

        data.generate_instance_with_gaz(test_file,'test')
        decode_results = load_model_decode(model_dir, data, 'test', gpu, dataset, seg)

    elif status == 'decode':       
        data = load_data_setting(dset_dir)
        data_build_gold(data,train_gold,dev_gold,test_gold)
        data.generate_instance_with_gaz(raw_file,'raw')
        decode_results = load_model_decode(model_dir, data, 'raw', gpu, seg)
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print("Please make sure all the arguments are valid!")




