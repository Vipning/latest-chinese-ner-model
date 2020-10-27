# -*- coding: utf-8 -*-

import sys
import numpy as np
from opencc import OpenCC
import json
import re
from utils.alphabet import Alphabet
# from transformers.tokenization_bert import BertTokenizer
NULLKEY = "-null-"

def get_word_label_list(line):
    cc = OpenCC('t2s')
    text = list(cc.convert(line['text']))#简体
    text = ['0' if c.isdigit() else c for c in text]#数字为0
    chars = text
    bigrams = [c1 + c2 for c1, c2 in zip(text, text[1:] + ['<eos>'])]
    label_entities = line.get('label', None)
    labels = ['O'] * len(chars)
    if label_entities is not None:
        for key, value in label_entities.items():
            for sub_name, sub_index in value.items():  # list
                for start_index, end_index in sub_index:
                    if start_index == end_index:
                        labels[start_index] = 'S-' + key  # 单字
                    else:
                        labels[start_index] = 'B-' + key
                        labels[end_index] = 'E-' + key
                        labels[start_index + 1:end_index] = ['M-' + key] * (end_index - start_index - 1)

    return chars,bigrams,labels

# def normalize_word(word):
#     new_word = ""
#     for char in word:
#         if char.isdigit():
#             new_word += '0'
#         else:
#             new_word += char
#     return new_word


def read_instance_with_gaz(input_file, gaz,word_alphabet,biword_alphabet,biword_count,gaz_alphabet, gaz_count, gaz_split, label_alphabet, number_normalized, max_sent_length):

    in_lines = open(input_file,'r',encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    labels = []
    word_Ids = []
    biword_Ids = []
    label_Ids = []

    for idx in range(len(in_lines)):
        line = in_lines[idx]
        line = json.loads(line)
        chars, bigrams, tags = get_word_label_list(line)
        for bigram in bigrams:
            biwords.append(bigram)
            biword_index = biword_alphabet.get_index(bigram)
            biword_Ids.append(biword_index)
        for label in tags:
            labels.append(label)
            label_Ids.append(label_alphabet.get_index(label))
        for char in chars:
            words.append(char)
            word_Ids.append(word_alphabet.get_index(char))

        if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
            gaz_Ids = []
            layergazmasks = []
            # gazchar_masks = []
            w_length = len(words)

            #[seq_len,4]
            gazs = [ [[] for i in range(4)] for _ in range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
            gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]

            max_gazlist = 0
            # max_gazcharlen = 0
            for idx in range(w_length):

                matched_list = gaz.enumerateMatchList(words[idx:])
                matched_length = [len(a) for a in matched_list]
                matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]

                for w in range(len(matched_Id)):
                    # g = matched_list[w]

                    if matched_length[w] == 1:  ## Single
                        gazs[idx][3].append(matched_Id[w])
                        gazs_count[idx][3].append(1)#？
                    else:
                        gazs[idx][0].append(matched_Id[w])   ## Begin
                        gazs_count[idx][0].append(gaz_count[matched_Id[w]])
                        wlen = matched_length[w]
                        gazs[idx+wlen-1][2].append(matched_Id[w])  ## End
                        gazs_count[idx+wlen-1][2].append(gaz_count[matched_Id[w]])
                        for l in range(wlen-2):
                            gazs[idx+l+1][1].append(matched_Id[w])  ## Middle
                            gazs_count[idx+l+1][1].append(gaz_count[matched_Id[w]])

                for label in range(4):#空
                    if not gazs[idx][label]:
                        gazs[idx][label].append(0)
                        gazs_count[idx][label].append(1)

                    max_gazlist = max(len(gazs[idx][label]),max_gazlist)

                matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]  #词号
                if matched_Id:
                    gaz_Ids.append([matched_Id, matched_length])
                    # print(gaz_Ids)
                else:
                    gaz_Ids.append([])

            ## batch_size = 1
            for idx in range(w_length):
                gazmask = []
                # gazcharmask = []

                for label in range(4):
                    label_len = len(gazs[idx][label])
                    count_set = set(gazs_count[idx][label])
                    if len(count_set) == 1 and 0 in count_set:
                        gazs_count[idx][label] = [1]*label_len

                    mask = label_len*[0]
                    mask += (max_gazlist-label_len)*[1]#1是遮蔽？

                    gazs[idx][label] += (max_gazlist-label_len)*[0]  ## padding
                    gazs_count[idx][label] += (max_gazlist-label_len)*[0]  ## padding

                    gazmask.append(mask)

                layergazmasks.append(gazmask)

            instence_texts.append([words, biwords,gazs, labels])
            instence_Ids.append([word_Ids, biword_Ids, gaz_Ids, label_Ids, gazs, gazs_count,layergazmasks])

            words = []
            biwords = []
            labels = []
            word_Ids = []
            biword_Ids = []
            label_Ids = []

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
    pretrain_emb[0,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word_alphabet.instance2index.items():
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
    with open(embedding_path, 'r',encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

