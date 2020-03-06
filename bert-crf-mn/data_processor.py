# encoding=utf8
import json
import tokenization
import config
import gensim
import os
from common import load_pickle
import numpy as np

class CluenerProcessor:
    def __init__(self,data_dir,vocab_file,label2id,max_seq_len):
        self.data_dir =data_dir
        self.vocab_file=vocab_file
        self.label2id=label2id
        self.max_seq_len=max_seq_len
        self.tokenizer=tokenization.FullTokenizer(vocab_file=vocab_file)


    
    def get_label_embedding(self,vocab=None,pretrained_label_embedding_file=None,output_file=None,embedding_dim=304):
        if not os.path.exists(output_file):
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_label_embedding_file,binary=False,
                                                    unicode_errors='ignore')
            scale = np.sqrt(3.0 / 300)
            text_wordvec=np.random.uniform(-scale,scale,size=(len(vocab),embedding_dim))
            for word, word_index in vocab.items():
                try:
                    word_vec = word2vec[word[2:]]
                    text_wordvec[word_index,:300] = word_vec
                    if word[0]=='B':
                      text_wordvec[word_index,300:]=np.asarray([0,1,0,0],dtype='float32')
                    elif word[0]=='I':
                      text_wordvec[word_index,300:]=np.asarray([0,0,1,0],dtype='float32')
                    elif word[0]=='S':
                      text_wordvec[word_index,300:]=np.asarray([0,0,0,1],dtype='float32')
                except Exception as e:
                    print(e)
                    print("exception:{}".format(word))
                    if word[0]=='O':
                      text_wordvec[word_index,300:] = np.asarray([1,0,0,0],dtype='float32')
                    continue
            self.label_embedding=text_wordvec
            np.save(output_file, text_wordvec)
        else:
            self.label_embedding=np.load(output_file, allow_pickle=True)

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "train.json"), "train")  # list

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "dev.json"), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "test.json"), "test")

    def _create_examples(self, input_path, mode):
        examples = []
        with open(input_path, 'r') as f:
            idx = 0
            for line in f:
                tokens = []
                labels = []
                json_d = {}
                line = json.loads(line.strip())
                textlist = list(line['text'])
                label_entities = line.get('label', None)
                labellist = ['O'] * len(textlist)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():  # list
                            for start_index, end_index in sub_index:
                                if start_index == end_index:
                                    labellist[start_index] = 'S-' + key  # 单字
                                else:
                                    labellist[start_index] = 'B-' + key
                                    labellist[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)

                for i,word in enumerate(textlist):
                    token = self.tokenizer.tokenize(word)
                    tokens.extend(token)
                    label_1 = labellist[i]
                    for m in range(len(token)):
                        if m == 0:
                            labels.append(label_1)
                        else:
                            print("some unknown token...")
                            labels.append(labels[0])
                assert len(tokens) < self.max_seq_len
                ntokens = []
                segment_ids = []
                label_ids = []
                ntokens.append("[CLS]")  # 句子开始设置CLS 标志
                segment_ids.append(0)
                label_ids.append(0)  # label2id["[CLS]"]
                for i, token in enumerate(tokens):
                    ntokens.append(token)
                    segment_ids.append(0)
                    label_ids.append(self.label2id[labels[i]])
                ntokens.append("[SEP]")
                segment_ids.append(0)
                # append("O") or append("[SEP]") not sure!
                label_ids.append(0)  # label2id["[SEP]"]
                input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
                input_len = len(input_ids)
                input_mask = [1] * len(input_ids)
                while len(input_ids) < self.max_seq_len:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    label_ids.append(0)
                    ntokens.append("**NULL**")
                assert len(input_ids) == self.max_seq_len
                assert len(input_mask) == self.max_seq_len
                assert len(segment_ids) == self.max_seq_len
                assert len(label_ids) == self.max_seq_len

                json_d['id'] = f"{mode}_{idx}"
                json_d['input_ids'] = input_ids
                json_d['input_mask'] = input_mask
                json_d['segment_ids'] = segment_ids
                json_d['label_ids'] = label_ids
                json_d['input_len'] = input_len
                idx += 1
                examples.append(json_d)
        return examples

    #!
    def get_aug_examples(self, distance_path, aug_num, mode):
        new_examples = []
        old_examples = []
        old_train = self.get_train_examples()
        if mode == 'train':
            old_examples = old_train
        elif mode == 'dev':
            old_examples = self.get_dev_examples()
        elif mode == 'test':
            old_examples = self.get_test_examples()

        examples_train = load_pickle(distance_path)

        for i, ele in enumerate(old_examples):
            cur_train = {}
            cur_train['ori_sentence'] = ele
            cur_train['aux_sentences'] = []
            sort_list = examples_train[i]
            sort_id = 0
            sort_id_list = []
            while len(cur_train['aux_sentences']) < aug_num:
                sort_sentence_id = sort_list[sort_id]
                if old_train[sort_sentence_id]['id'] != old_examples[i]['id']:#不同句子
                    cur_train['aux_sentences'].append(old_train[sort_sentence_id].copy())
                    sort_id_list.append(sort_list[sort_id])
                sort_id += 1
                if sort_id >= len(sort_list):
                    raise ValueError('Need more sentences id!')

            new_examples.append(cur_train)

        return new_examples