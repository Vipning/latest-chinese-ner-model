import math
import random

class BatchManager(object):

    def __init__(self,data,batch_size,vocab,label2id,shuffle=True):
        self.data = data
        self.shuffle = shuffle
        self.batch_size=batch_size
        self.vocab = vocab
        self.label2id = label2id
        self.reset()

    def reset(self):
        data=self.preprocess(self.data)
        self.batch_data = self.shuffle_and_pad(data,self.batch_size,self.shuffle)
        self.len_data = len(self.batch_data)

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            text_a = d['context']
            tokens = [self.vocab.to_index(w) for w in text_a.split(" ")]
            x_len = len(tokens)
            text_tag = d['tag']
            tag_ids = [self.label2id[tag] for tag in text_tag.split(" ")]
            processed.append((tokens, tag_ids, x_len, text_a, text_tag))

        return processed

    def shuffle_and_pad(self, data, batch_size,shuffle=True):
        num_batch = int(math.ceil(len(data) /batch_size))
        shuffle_data=data
        if shuffle:
            random.shuffle(shuffle_data)
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(shuffle_data[i*batch_size:(i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        #tokens, tag_ids, x_len, text_a, text_tag
        input_ids = []
        labels_ids = []
        input_mask = []
        input_lens = []
        max_length = max([sentence[2] for sentence in data])
        for line in data:
            tokens,tag_ids,x_len,_,_ = line
            padding = [0] * (max_length - len(tokens))
            input_ids.append(tokens + padding)
            labels_ids.append(tag_ids + padding)
            input_lens.append(x_len)
            mask_ = [0] * max_length
            mask_[:len(tokens)] = [1]*len(tokens)
            input_mask.append(mask_)
        return [input_ids,input_mask,labels_ids,input_lens]

    def iter_batch(self, shuffle=True):
        if shuffle:
            random.shuffle(self.batch_data)#batch间打乱
        for idx in range(self.len_data):
            yield self.batch_data[idx]