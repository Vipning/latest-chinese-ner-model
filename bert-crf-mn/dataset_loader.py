import math
import random

class BatchManager(object):

    def __init__(self,data,batch_size,shuffle=True):
        self.data = data
        self.shuffle = shuffle
        self.batch_size=batch_size
        self.reset()

    def reset(self):
        data=self.preprocess(self.data)
        self.batch_data = self.shuffle_batch(data,self.batch_size,self.shuffle)
        self.len_data = len(self.batch_data)

    def convert(self,d):
        input_ids = d['input_ids']
        input_mask = d['input_mask']
        segment_ids = d['segment_ids']
        label_ids = d['label_ids']
        input_len = d['input_len']
        return (input_ids,label_ids,input_len,input_mask,segment_ids)

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            ele = {}
            ele['ori'] = self.convert(d['ori_sentence'])
            ele['aug'] = []
            for s in d['aux_sentences']:
                ele['aug'].append(self.convert(s))

            processed.append(ele)

        return processed

    def shuffle_batch(self, data, batch_size,shuffle=True):
        num_batch = int(math.ceil(len(data) /batch_size))
        if shuffle:
            random.shuffle(data)
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.batchify(data[i*batch_size:(i+1)*batch_size]))

        return batch_data


    def aug_batchify(self,data):
        aug_input_ids = []
        aug_label_ids = []
        aug_input_mask = []
        aug_input_lens = []
        aug_segment_ids=[]
        for sentence in data:
            input_ids = []
            input_mask = []
            label_ids = []
            input_lens = []
            segment_ids = []
            for line in sentence['aug']:
                input_ids.append(line[0])
                label_ids.append(line[1])
                input_lens.append(line[2])
                input_mask.append(line[3])
                segment_ids.append(line[4])
            aug_input_ids.append(input_ids)
            aug_input_mask.append(input_mask)
            aug_label_ids.append(label_ids)
            aug_input_lens.append(input_lens)
            aug_segment_ids.append(segment_ids)
        return [aug_input_ids, aug_input_mask, aug_label_ids, aug_input_lens,aug_segment_ids]

    def ori_batchify(self,data):
        input_ids =  []
        label_ids = []
        input_mask = []
        input_lens = []
        segment_ids =[]
        for sentence in data:
            line=sentence['ori']
            input_ids.append(line[0])
            label_ids.append(line[1])
            input_lens.append(line[2])
            input_mask.append(line[3])
            segment_ids.append(line[4])

        return [input_ids, input_mask, label_ids, input_lens,segment_ids]

    def batchify(self,data):
        res = {
            'ori': self.ori_batchify(data),
            'aug': self.aug_batchify(data)
        }
        return res

    def iter_batch(self, shuffle=True):

        if shuffle:
            random.shuffle(self.batch_data)#batch间打乱
        for idx in range(self.len_data):
            yield self.batch_data[idx]