import numpy as np
import math
import sys
import os
import re
from .scorer.event_scorer import *
import json

def strmatrix2list(matrix,sid,chr_ids=[]):
    res = []
    for trigger in matrix:

        if ',' not in trigger:
            indx2 = trigger.index(']')
            beg,end,tag = int(trigger[1:indx2]),int(trigger[1:indx2]),trigger[indx2+1:]
        else:
            indx = trigger.index(',')
            indx2 = trigger.index(']')
            beg,end,tag = int(trigger[1:indx]),int(trigger[indx+1:indx2]),trigger[indx2+1:]
        if chr_ids:
            offset = chr_ids[beg]
            length = chr_ids[end] - chr_ids[beg] + 1
        else:
            offset = beg
            length = end - beg + 1
        res.append((str(sid),offset,length,tag))
    return res


def get_ner_fmeasure(gold_dict, golden_lists, predict_lists, label_type, use_scorer):
    fw = open('ace_mg.res','w',encoding='utf-8')
    sent_num = len(predict_lists)
    golden_full = []
    predict_full = []
    right_tag = 0
    right_span_tag = 0
    all_tag = 0
    cursids = set()
    for idx in range(0,sent_num):
        sid,predict_list = predict_lists[idx]
        if isinstance(sid,list):
            sid, chr_ids = sid
        else:
            chr_ids = []
        cursids.add(sid)
        golden_list = golden_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy][0] == predict_list[idy][0]:
                right_span_tag += 1
                if predict_list[idy] == golden_list[idy]:
                    right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        gold_matrix2 = strmatrix2list(gold_matrix,sid,chr_ids)
        pred_matrix2 = strmatrix2list(pred_matrix,sid,chr_ids)
        golden_full += gold_matrix2
        predict_full += pred_matrix2

    right_span = 0
    right_type = 0
    pred_dict = {}
    
    for sid, offset, length, tag in predict_full:
        key = (sid, offset, length)    
        if key in gold_dict:
            right_span += 1
            if tag in gold_dict[key]:
                right_type += 1
        fw.write('\t'.join([sid,str(offset),str(length),tag])+'\n')
        pred_dict[key] = [tag]

    gold_dict2 = {}
    dict2_sid = set()
    for sid, offset, length, tag in golden_full:
        key = (sid, offset, length)
        if key not in gold_dict2:
            gold_dict2[key] = []
        gold_dict2[key].append(tag)
        dict2_sid.add(sid)

    golden_full = 0
    del_key = []
    for sid, offset, length in gold_dict:
        if sid in cursids:
            golden_full += 1
        else:
            print('SID miss:'+sid) 
        if sid not in dict2_sid:
  
            del_key.append((sid,offset,length))
            print('del:',(sid,offset,length)) 
    

    print(golden_full,len(gold_dict))
    golden_full = len(gold_dict)

    if use_scorer:
        print('Using KBP scorer tooklit')
        pred_strs = transform_to_score_list(pred_dict)
        gold_strs = transform_to_score_list(gold_dict)
        eval_rst = score(gold_strs,pred_strs)[2]
        p_span,r_span,f_span = eval_rst['plain']['micro']
        p_type,r_type,f_type = eval_rst['mention_type']['micro']
    else:
        p_span = 100.0 * right_span / len(pred_dict) if len(pred_dict) > 0 else -1
        r_span = 100.0 * right_span / golden_full if golden_full > 0 else -1
        f_span = p_span * r_span * 2 / (p_span + r_span) if p_span + r_span > 0 else -1

        p_type = 100.0 * right_type / len(pred_dict) if len(pred_dict) > 0 else -1
        r_type = 100.0 * right_type / golden_full if golden_full > 0 else -1
        f_type = p_type * r_type * 2 / (p_type + r_type) if p_type + r_type > 0 else -1

        print(right_type,len(predict_full),golden_full,right_span)

    acc_span = 100.0*(right_span_tag+0.0)/all_tag
    acc_type = 100.0*(right_tag+0.0)/all_tag
    
    return acc_span, p_span, r_span, f_span, acc_type, p_type, r_type, f_type

def get_ner_fmeasure_old(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):

        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)

        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    
    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return -1, -1, -1, -1, accuracy, precision, recall, f_measure

'''
## input as sentence level labels
def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES",use_scorer=True):
    def get_str_dict(mp):
        res = {}
        for k,v in mp.items():
            key = '#'.join([str(x) for x in k])
            res[key] = v
        return res
    #fw = open('debug2.out','w',encoding='utf-8')
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_tag = 0
    right_span_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy][0] == predict_list[idy][0]:
                right_span_tag += 1
                if predict_list[idy] in golden_list[idy].split('#'):
                    right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list) # ['[beg,end]tag',...]
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        gold_matrix2 = strmatrix2list(gold_matrix,idx)
        pred_matrix2 = strmatrix2list(pred_matrix,idx)
        golden_full += gold_matrix2
        predict_full += pred_matrix2
    
    right_span = 0
    right_type = 0
    gold_dict = {}
    pred_dict = {}
    for sid, offset, length, tag in golden_full:
        key = (sid, offset, length)
        if key not in gold_dict:
            gold_dict[key] = []
        tag = tag.split('#')
        for ti,tg in enumerate(tag):            
            if len(tg)>2 and tg[1] == '-' and tg[0] in ['B','I','M','E','S']:
                tag[ti] = tg[2:] # 由于多标签只去除了前面的B-等等
        gold_dict[key] += tag
    for sid, offset, length, tag in predict_full:
        key = (sid, offset, length)
        cur_sent = []
        if key in gold_dict:
            right_span += 1
            if tag in gold_dict[key]:
                right_type += 1
        pred_dict[key] = [tag]

    #fw.write(json.dumps([get_str_dict(gold_dict),get_str_dict(pred_dict)]))


    if use_scorer:
        pred_strs = transform_to_score_list(pred_dict)
        gold_strs = transform_to_score_list(gold_dict)
        eval_rst = score(gold_strs,pred_strs)[2]
        p_span,r_span,f_span = eval_rst['plain']['micro']
        p_type,r_type,f_type = eval_rst['mention_type']['micro']
    else:
        p_span = 1.0 * right_span / len(predict_full) if len(predict_full) > 0 else -1
        r_span = 1.0 * right_span / len(golden_full) if len(golden_full) > 0 else -1
        f_span = p_span * r_span * 2 / (p_span + r_span) if p_span + r_span > 0 else -1

        p_type = 1.0 * right_type / len(predict_full) if len(predict_full) > 0 else -1
        r_type = 1.0 * right_type / len(golden_full) if len(golden_full) > 0 else -1
        f_type = p_type * r_type * 2 / (p_type + r_type) if p_type + r_type > 0 else -1

        print(right_type,len(predict_full),len(golden_full),right_span)

    acc_span = (right_span_tag+0.0)/all_tag
    acc_type = (right_tag+0.0)/all_tag
    
    return acc_span, p_span, r_span, f_span, acc_type, p_type, r_type, f_type
'''
def get_ner_fmeasure_pre(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = {}
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):

        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)

        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full[idx] = right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag

    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return right_full

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):

    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):

        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i) 
            index_tag = current_label.replace(begin_label,"",1)            
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        if ',' not in whole_tag:
            whole_tag += ',' + str(list_len-1)
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)

    return stand_matrix


def get_ner_BIO(label_list):

    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-' 
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag 
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        if ',' not in whole_tag:
            whole_tag += ',' + str(list_len-1)
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix

def readSentence(input_file):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences,labels

def readTwoLabelSentence(input_file, pred_col=-1):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if "##score##" in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[1])
            predict_label.append(pair[pred_col])
            
    return sentences,golden_labels,predict_labels

def fmeasure_from_file(golden_file, predict_file, label_type="BMES"):
    print("Get f measure from file:", golden_file, predict_file)
    print("Label format:",label_type)
    golden_sent,golden_labels = readSentence(golden_file)
    predict_sent,predict_labels = readSentence(predict_file)
    acc, P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print("Acc:%s, P:%s R:%s, F:%s"%(acc, P,R,F))

def fmeasure_from_singlefile(twolabel_file, label_type="BMES", pred_col=-1):
    sent,golden_labels,predict_labels = readTwoLabelSentence(twolabel_file, pred_col)
    P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print("P:%s, R:%s, F:%s"%(P,R,F))

def read_output(fn):
    res = dict()
    fr = open(fn,'r',encoding='utf-8')
    for line in fr:
        if len(line)<3:
            continue
        line = line.replace("(u'","('")
        key,val = line.strip().split(': ')
        id,offset,length = key[1:-1].replace("'","").replace(" ","").split(',')
        val = val[1:-1].replace("'","").replace(" ","").split(',')
        offset = int(offset)
        length = int(length)
        res[(id,offset,length)] = val
    return res
        

if __name__ == '__main__':
    
    gold = [['O','B-AA','I-AA','O','O','B-CC','B-DD','O']]
    pred = [['id1',['O','B-CC','I-CC','O','O','B-AA','B-DD','I-DD']]]
    gold_dict = {('id1',1,2):['AA'],('id1',5,1):['CC','AA'],('id1',6,1):['DD']}
    print(get_ner_fmeasure(gold_dict, gold, pred, label_type="BIO",use_scorer=False))

