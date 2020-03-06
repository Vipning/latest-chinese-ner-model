# encoding = utf8
import numpy as np
import tensorflow as tf
from nn_utils import shape,CustomLSTMCell,projection
import optimization
import modeling
import math
from attention import multihead_attention,attention_bias

class NERModel(object):
    def __init__(self, config):

        self.config = config
        self.max_seq_len=config['max_seq_len']
        self.label2id = config['label2id']
        self.num_tags = len(self.label2id)
        self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
        # add placeholders for the model
        self.input_ids = tf.placeholder(dtype=tf.int32,
                                        shape=[None,None],
                                        name="Input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="Input_mask")
        self.labels_ids = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="Labels_ids")
        self.input_lens = tf.placeholder(dtype=tf.int32,
                                         shape=[None],
                                         name="Input_lens")
        self.segment_ids = tf.placeholder(dtype=tf.int32,
                                         shape=[None,None],
                                         name="Segment_ids")
        self.a_input_ids = tf.placeholder(dtype=tf.int32,
                                        shape=[None,None,None],
                                        name="Aug_input_ids")
        self.a_input_mask = tf.placeholder(dtype=tf.int32,
                                         shape=[None,None,None],
                                         name="Aug_input_mask")
        self.a_labels_ids = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None,None],
                                         name="Aug_labels_ids")
        self.a_input_lens = tf.placeholder(dtype=tf.int32,
                                         shape=[None,None],
                                         name="Aug_input_lens")
        self.a_segment_ids = tf.placeholder(dtype=tf.int32,
                                          shape=[None,None,None],
                                          name="Segment_ids")
        self.is_train = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')

        self.logits=self.get_predictions(self.input_ids,self.input_mask,self.input_lens,self.segment_ids,self.a_input_ids,self.a_labels_ids,self.a_input_mask,self.a_input_lens,self.a_segment_ids,self.is_train)
        self.loss, self.trans = self.loss_layer(self.logits,self.labels_ids,self.input_lens)

        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, self.config['tf_checkpoint'])
        tf.train.init_from_checkpoint(self.config['init_checkpoint'], assignment_map)
        initialized_vars = [v for v in tvars if v.name in initialized_variable_names]
        not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
        for v in initialized_vars:
            print('--initialized: %s, shape = %s' % (v.name, v.shape))
        for v in not_initialized_vars:
            print('--not initialized: %s, shape = %s' % (v.name, v.shape))

        num_train_steps = math.ceil(self.config['train_examples_len'] / self.config["batch_size"])* self.config["epochs"]
        num_warmup_steps = int(num_train_steps* self.config['warmup_proportion'])
        
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = optimization.create_custom_optimizer(tvars,self.loss,self.config['bert_learning_rate'],self.config['task_learning_rate'],
                                             num_train_steps,num_warmup_steps, False,self.global_step,freeze=-1,
                                             task_opt=self.config['task_optimizer'], eps=config['adam_eps'])

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def get_predictions(self,input_ids,input_mask,input_lens,segment_ids,a_input_ids,a_labels_ids,a_input_mask,a_input_lens,a_segment_ids,is_train):

        self.keep_prob = self.get_keep_rate(self.config['dropout_rate'],is_train)
        # self.lstm_keep_prob = self.get_keep_rate(self.config['lstm_dropout'],is_train)
        self.attention_keep_prob = self.get_keep_rate(self.config['attention_dropout'],is_train)

        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_train,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )
        output_layer = model.get_sequence_output()
        seq_length = shape(output_layer, -2)

        #辅助
        detect_batch_size = shape(a_input_ids, 0)
        detect_a_batch_size = shape(a_input_ids, 1)
        a_input_ids = tf.reshape(a_input_ids, [detect_batch_size * detect_a_batch_size, -1])
        a_input_mask = tf.reshape(a_input_mask, [detect_batch_size * detect_a_batch_size, -1])
        a_labels_ids = tf.reshape(a_labels_ids, [detect_batch_size * detect_a_batch_size, -1])
        a_segment_ids =tf.reshape(a_segment_ids,[detect_batch_size * detect_a_batch_size, -1])
        a_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_train,
            input_ids=a_input_ids,
            input_mask=a_input_mask,
            token_type_ids=a_segment_ids,
            use_one_hot_embeddings=False
        )
        a_output_layer=a_model.get_sequence_output()

        a_labels_emb = self.label_embedding_layer(a_labels_ids,True)
        label_emb_size = shape(a_labels_emb, -1)
        a_labels_emb = tf.reshape(a_labels_emb, [detect_batch_size, detect_a_batch_size, -1, label_emb_size])

        a_hidden_size = shape(a_output_layer, -1)
        a_hidden_input = tf.reshape(a_output_layer, [detect_batch_size, detect_a_batch_size, -1, a_hidden_size])

        hidden_input = tf.expand_dims(output_layer, 1)
        a_hidden_input = tf.transpose(a_hidden_input, [0, 1, 3, 2])
        temp_feature = tf.matmul(hidden_input, a_hidden_input)
        prob_feature = tf.nn.softmax(temp_feature)
        aug_represent = tf.matmul(prob_feature, a_labels_emb)
        aug_represent = tf.reduce_mean(aug_represent, axis=1)

        final_represent = tf.concat([output_layer, aug_represent], 2)
        final_represent = tf.nn.dropout(final_represent,keep_prob=self.keep_prob)
        
        attention_outputs = self.self_attention(final_represent,input_mask,keep_prob=self.attention_keep_prob)
        
        with tf.variable_scope("logits"):

            final_size=shape(attention_outputs,-1)

            output_weight = tf.get_variable(
                "output_weights", [self.num_tags,final_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            output_bias = tf.get_variable(
                "output_bias", [self.num_tags], initializer=tf.zeros_initializer()
            )
            output_layer = tf.reshape(attention_outputs, [-1, final_size])
            logits = tf.matmul(output_layer,output_weight, transpose_b=True)
            logits = tf.reshape(logits,[-1, seq_length, self.num_tags])

            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, shape=(-1, seq_length, self.num_tags))

            return logits


    # 是否训练
    def get_keep_rate(self, dropout_rate, is_training):
        return 1.0 - (tf.cast(is_training, tf.float32) * dropout_rate)

    def label_embedding_layer(self, label_inputs, pretrain_label_embedding=False):

        with tf.variable_scope('label_embedding',reuse=tf.AUTO_REUSE):
            if pretrain_label_embedding:
                print('load pretrain label embedding')
                label_lookup = tf.get_variable(name="label_embedding", shape=self.config['label_embedding'].shape,
                                               initializer=tf.constant_initializer(self.config['label_embedding']))
            else:
                scale = np.sqrt(3.0 / 300)
                label_lookup = tf.get_variable(name='label_embedding',
                                               shape=[len(self.label2id), self.config['label_embedding_size']],
                                               initializer=tf.random_uniform_initializer(-scale,scale))

        embed = tf.nn.embedding_lookup(label_lookup, label_inputs)

        return embed
    
    def self_attention(self,x,input_mask,keep_prob):
      attn_bias = attention_bias(tf.cast(input_mask,tf.float32))
      with tf.variable_scope("self_attention"):
          y = multihead_attention(
              x,
              None,
              attn_bias,
              self.config['key_size'],
              self.config['value_size'],
              self.config['sa_output_size'],
              self.config['num_heads'],
              keep_prob,
              attention_function='dot_product'
          )
      return y

    def biLSTM_layer(self, lstm_inputs,lstm_dim,lengths,num_layers,keep_prob=1.):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        batch_size=shape(lstm_inputs,0)
        with tf.variable_scope("BiLSTM"):
            for layer in range(num_layers):
              with tf.variable_scope("layer_{}".format(layer)):
                with tf.variable_scope("forward"):
                    cell_fw = CustomLSTMCell(lstm_dim,batch_size,keep_prob)
                with tf.variable_scope("backward"):
                    cell_bw = CustomLSTMCell(lstm_dim,batch_size,keep_prob)
                state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [batch_size, 1]),
                                                        tf.tile(cell_fw.initial_state.h, [batch_size, 1]))
                state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [batch_size, 1]),
                                                        tf.tile(cell_bw.initial_state.h, [batch_size, 1]))

                (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=lstm_inputs,
                    sequence_length=lengths,
                    initial_state_fw=state_fw,
                    initial_state_bw=state_bw
                    )
                text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.nn.dropout(text_outputs,keep_prob)
                if layer > 0:
                  highway_gates = tf.sigmoid(projection(text_outputs,shape(text_outputs, 2))) # [num_sentences, max_sentence_length, emb]
                  text_outputs = highway_gates * text_outputs + (1 - highway_gates) * lstm_inputs
                lstm_inputs=text_outputs
                
            return lstm_inputs

    def loss_layer(self, project_logits, labels_ids,lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        # batch_size = shape(project_logits, 0)
        # num_steps = shape(project_logits, 1)
        with tf.variable_scope("crf_loss" if not name else name):

            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=project_logits,
                tag_indices=labels_ids,
                sequence_lengths=lengths)

            return tf.reduce_mean(-log_likelihood), trans

    def create_feed_dict(self,instance,is_predict):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        #input_ids, input_mask, labels_ids, input_lens,segment_ids,is_train
        batch,is_train=instance
        input_ids, input_mask, labels_ids, input_lens,segment_ids = batch['ori']
        a_input_ids, a_input_mask, a_labels_ids, a_input_lens,a_segment_ids = batch['aug']
        feed_dict = {
            self.input_ids: np.asarray(input_ids),
            self.input_mask: np.asarray(input_mask),
            self.input_lens: np.asarray(input_lens),
            self.segment_ids:np.asarray(segment_ids),
            self.a_input_ids: np.asarray(a_input_ids),
            self.a_input_mask: np.asarray(a_input_mask),
            self.a_labels_ids: np.asarray(a_labels_ids),
            self.a_input_lens: np.asarray(a_input_lens),
            self.a_segment_ids:np.asarray(a_segment_ids),
            self.is_train: np.asarray(is_train)

        }

        if not is_predict:
            feed_dict[self.labels_ids] = np.asarray(labels_ids)

        return feed_dict

    def run_step(self,sess,instance,is_train,is_predict=False):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(instance,is_predict)
        if is_train:#train
            global_step,loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            if is_predict:#predict
                logits, lengths = sess.run([self.logits, self.input_lens], feed_dict)

                return logits, lengths
            else:#dev
                logits, loss, lengths = sess.run([self.logits, self.loss, self.input_lens], feed_dict)

                return logits, loss, lengths

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        paths = []
        for score, length in zip(logits, lengths):
            logits = score[:length]
            path, _ = tf.contrib.crf.viterbi_decode(logits, matrix)

            paths.append(path)
        return paths

    def evaluate(self, sess, data_manager, metric):
        # results = []
        eval_loss = []
        trans = self.trans.eval()  # 转移矩阵
        for example in data_manager.iter_batch():
            _,_, input_tags, input_lens,_= example['ori']
            instance = (example,False)
            scores, loss, lengths = self.run_step(sess,instance,False)
            batch_paths = self.decode(scores,lengths, trans)  # 要去padding
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=batch_paths, label_paths=target)
            eval_loss.append(loss)

        return metric.result(), np.mean(eval_loss)

    def evaluate_line(self, sess, inputs):
        trans = self.trans.eval()
        instance = (inputs, False)
        scores, lengths = self.run_step(sess,instance, False,True)
        batch_paths = self.decode(scores, lengths, trans)
        return batch_paths