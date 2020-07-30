# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from relative_transformer_module import relative_multi_head_attention,feedforward
import optimization
import math
from common import batchify_with_label
from nn_utils import shape

class NERModel(object):
    def __init__(self,config,data):

        self.config = config
        self.data=data
        self.num_tags=data.label_alphabet_size
        self.gaz_emb_dim = data.gaz_emb_dim
        self.word_emb_dim = data.word_emb_dim
        self.biword_emb_dim = data.biword_emb_dim
        # 参数初始化
        self.initializer = initializers.xavier_initializer()
        # add placeholders for the model
        self.is_train=tf.placeholder(dtype=tf.bool,shape=[],name='is_train')

        self.word_inputs = tf.placeholder(dtype=tf.int32,
                                        shape=[None, None],
                                        name="word_inputs")

        self.biword_inputs=tf.placeholder(dtype=tf.int32,shape=[None,None],
                                          name='biword_inputs')

        self.mask =tf.placeholder(dtype=tf.int32,
                                  shape=[None,None],
                                  name='mask')

        self.word_seq_lengths=tf.placeholder(dtype=tf.int32,
                                        shape=[None],
                                        name="word_seq_lengths")

        self.batch_label=tf.placeholder(dtype=tf.int32,
                                        shape=[None, None],
                                        name="batch_label")

        self.layer_gaz=tf.placeholder(dtype=tf.int32,
                                        shape=[None, None,4,None],
                                        name="layer_gaz")
        self.gaz_mask_input = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None, 4, None],
                                             name="gaz_mask_input")

        self.gaz_count=tf.placeholder(dtype=tf.int32,
                                        shape=[None, None,4,None],
                                        name="gaz_count")


        self.embedding_keep_prob = self.get_keep_rate(self.config['embedding_dropout'], self.is_train)
        self.fc_keep_prob = self.get_keep_rate(self.config['fc_dropout'], self.is_train)
        self.attention_keep_prob = self.get_keep_rate(self.config['attention_dropout'], self.is_train)
        self.ffnn_keep_prob = self.get_keep_rate(self.config['ffnn_dropout'], self.is_train)

        batch_size = shape(self.word_inputs,0)
        seq_len=shape(self.word_inputs,1)

        #word
        word_embs = self.word_embedding_layer(self.word_inputs,data.word_alphabet.size(),self.word_emb_dim)
        word_embs = word_embs*tf.expand_dims(tf.cast(self.mask,dtype=tf.float32),-1)
        #biword
        biword_embs = self.biword_embedding_layer(self.biword_inputs,data.biword_alphabet.size(),self.biword_emb_dim)
                                           
        biword_embs = biword_embs * tf.expand_dims(tf.cast(self.mask,dtype=tf.float32), -1)

        word_inputs_d = tf.concat([word_embs,biword_embs],-1)
        word_inputs_d = tf.nn.dropout(word_inputs_d,self.embedding_keep_prob)
        #gaz
        gaz_embeds = self.gaz_embedding_layer(self.layer_gaz,data.gaz_alphabet.size(),self.gaz_emb_dim)
        gaz_embeds = tf.nn.dropout(gaz_embeds,self.embedding_keep_prob)
        gaz_embeds = gaz_embeds * (1.0-tf.expand_dims(tf.cast(self.gaz_mask_input,dtype=tf.float32), -1))#
        #dropout

        count_sum = tf.reduce_sum(self.gaz_count,3, keepdims = True)  #(b,l,4,gn)每个位置的单词总数
        count_sum = tf.reduce_sum(count_sum,2, keepdims = True)  #(b,l,1,1)#4个位置也要算？

        weights = tf.divide(self.gaz_count,count_sum)  #(b,l,4,g)tf.int32/tf.int32=tf.float64
        weights = weights*4
        weights = tf.cast(tf.expand_dims(weights,-1),tf.float32)
        gaz_embeds = weights*gaz_embeds  #(b,l,4,g,e)
        gaz_embeds = tf.reduce_sum(gaz_embeds,3)  #(b,l,4,e)

        gaz_embeds_cat = tf.reshape(gaz_embeds,(batch_size,seq_len,4*self.gaz_emb_dim))  #(b,l,4*ge) l=length

        word_input_cat = tf.concat([word_inputs_d,gaz_embeds_cat],-1)  #(b,l,we+4*ge)

        inputs = tf.layers.dense(word_input_cat,self.config['attention_size'], name='input_fc',
                                 kernel_initializer=self.initializer)

        outputs = self.adapting_transformer_layer(inputs,self.mask,self.config['ffnn_size'],
                                                  self.config['num_heads'], self.config['attn_blocks_num'],
                                                  self.attention_keep_prob, self.ffnn_keep_prob)
        # fc_dropout
        outputs = tf.nn.dropout(outputs, self.fc_keep_prob)
        # 分类
        self.logits = self.project_layer(outputs)
        # crf计算损失
        self.loss, self.trans = self.loss_layer(self.logits,self.word_seq_lengths)

        num_train_steps = math.ceil(self.config['train_examples_len'] / self.config["batch_size"]) * self.config[
            "epochs"]
        num_warmup_steps = int(num_train_steps * self.config['warmup_proportion'])
        self.global_step = tf.train.get_or_create_global_step()
        trainable_params = tf.trainable_variables()
        for var in trainable_params:
            print(" trainable_params name = %s, shape = %s" % (var.name, var.shape))

        self.train_op = optimization.create_optimizer(trainable_params, self.loss, self.config['other_learning_rate'],
                                                      self.config['crf_learning_rate'], num_train_steps,
                                                      num_warmup_steps, self.global_step)
        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    #是否训练
    def get_keep_rate(self, dropout_rate, is_training):
        return 1.0 - (tf.cast(is_training,tf.float32) * dropout_rate)

    #嵌入层
    def word_embedding_layer(self,inputs,vocab_size,embedding_size,pretrain_embedding=True):
        with tf.variable_scope('word_embedding'):
            if pretrain_embedding:
                W = tf.get_variable(name="W", shape=self.data.pretrain_word_embedding.shape, initializer=tf.constant_initializer(self.data.pretrain_word_embedding))
            else:
                W = tf.get_variable(name='W', shape=[vocab_size,embedding_size],
                                              initializer=self.initializer)

        embed = tf.nn.embedding_lookup(W,inputs)

        return embed

    def gaz_embedding_layer(self,inputs,vocab_size,embedding_size,pretrain_embedding=True):
        with tf.variable_scope('gaz_embedding'):
            if pretrain_embedding:
                W = tf.get_variable(name="W", shape=self.data.pretrain_gaz_embedding.shape, initializer=tf.constant_initializer(self.data.pretrain_gaz_embedding))
            else:
                W = tf.get_variable(name='W', shape=[vocab_size,embedding_size],
                                              initializer=self.initializer)

        embed = tf.nn.embedding_lookup(W,inputs)

        return embed

    def biword_embedding_layer(self,inputs,vocab_size,embedding_size,pretrain_embedding=True):
        with tf.variable_scope('biword_embedding'):
            if pretrain_embedding:
                W = tf.get_variable(name="W", shape=self.data.pretrain_biword_embedding.shape, initializer=tf.constant_initializer(self.data.pretrain_biword_embedding))
            else:
                W = tf.get_variable(name='W', shape=[vocab_size,embedding_size],
                                              initializer=self.initializer)

        embed = tf.nn.embedding_lookup(W,inputs)

        return embed

    def adapting_transformer_layer(self, batch_input, mask, ffnn_size, num_heads=8, attn_blocks_num=2,
                                   attention_keep_prob=1.0, ffnn_keep_prob=1.0):

        attn_outs = batch_input
        attention_size = shape(attn_outs, -1)
        for block_id in range(attn_blocks_num):
            with tf.variable_scope("num_blocks_{}".format(block_id)):
                attn_outs = relative_multi_head_attention(
                    attn_outs, mask, attention_size, num_heads, attention_keep_prob, reuse=False)
                attn_outs = feedforward(attn_outs, [ffnn_size, attention_size], ffnn_keep_prob, reuse=False)

        return attn_outs

    def project_layer(self, lstm_outputs):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        num_steps = shape(lstm_outputs, 1)
        hidden_size = shape(lstm_outputs, 2)
        with tf.variable_scope("project"):
            output = tf.reshape(lstm_outputs, shape=[-1, hidden_size])
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[hidden_size, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(output, W, b)

            return tf.reshape(pred, [-1, num_steps,self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags, self.num_tags],
                initializer=self.initializer)

            log_likelihood, trans = crf_log_likelihood(
                inputs=project_logits,
                tag_indices=self.batch_label,
                transition_params=trans,
                sequence_lengths=lengths)

            return tf.reduce_mean(-log_likelihood), trans

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        for score, length in zip(logits, lengths):
            logits = score[:length]
            path, _ = viterbi_decode(logits, matrix)
            paths.append(path)
        return paths

    #输入
    def create_feed_dict(self,batch,is_predict):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        batch_word,batch_biword,batch_wordlen,batch_label,layer_gaz,gaz_count, gaz_mask, mask,is_train=batch
        feed_dict = {
            self.word_inputs: np.asarray(batch_word),
            self.biword_inputs:np.asarray(batch_biword),
            self.word_seq_lengths: np.asarray(batch_wordlen),
            self.mask: np.asarray(mask),
            self.layer_gaz:np.asarray(layer_gaz),
            self.gaz_mask_input:np.asarray(gaz_mask),
            self.gaz_count:np.asarray(gaz_count),
            self.is_train:np.asarray(is_train)
        }
        if not is_predict:
            feed_dict[self.batch_label] = np.asarray(batch_label)

        return feed_dict

    def run_step(self, sess,batch,is_train,is_predict=False):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(batch,is_predict)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step,loss
        else:
            if is_predict:
                logits, lengths = sess.run([self.logits, self.word_seq_lengths], feed_dict)

                return logits, lengths
            else:
                logits, loss, lengths = sess.run([self.logits, self.loss, self.word_seq_lengths], feed_dict)

                return logits, loss, lengths

    def evaluate(self, sess,data_Ids,metric,batch_size):
        # results = []
        eval_loss = []
        trans = self.trans.eval()  # 转移矩阵
        # batch_size = args.batch_size
        train_num = len(data_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data_Ids[start:end]  # train_Ids
            if not instance:
                continue
            _,batch_word, batch_biword,batch_wordlen, batch_label, layer_gaz, gaz_count,gaz_mask, mask=batchify_with_label(instance)
            batch=(batch_word,batch_biword,batch_wordlen, batch_label, layer_gaz, gaz_count,gaz_mask, mask,False)
            scores, loss, lengths = self.run_step(sess,batch,False)
            batch_paths = self.decode(scores,batch_wordlen, trans)  # 要去padding
            target = [input_[:len_] for input_, len_ in zip(batch_label,batch_wordlen)]
            metric.update(pred_paths=batch_paths, label_paths=target)
            eval_loss.append(loss)

        return metric.result(), np.mean(eval_loss)

    def evaluate_line(self, sess, inputs):#inputs-->batch
        trans = self.trans.eval()
        scores, lengths = self.run_step(sess, inputs,False,True)
        batch_paths = self.decode(scores, lengths, trans)

        return batch_paths