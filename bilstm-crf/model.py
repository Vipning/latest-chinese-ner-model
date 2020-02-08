# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from rnncell import shape,CustomLSTMCell,projection


class NERModel(object):
    def __init__(self,config):

        self.config = config
        self.lstm_dim =config["lstm_dim"]
        self.vocab_size=config['vocab_size']
        self.embedding_size=config['embedding_size']
        self.keep_prob=config['keep_prob']
        self.label2id=config['label2id']
        self.num_tags=len(self.label2id)
        #参数初始化
        self.initializer = initializers.xavier_initializer()
        # add placeholders for the model
        #input_ids,input_mask,labels_ids,input_lens
        self.input_ids = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
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

        #字符嵌入
        embedding=self.embedding_layer(self.input_ids)
        #dropout
        lstm_inputs=self.spatial_dropout(embedding,self.keep_prob)
        lstm_inputs=lstm_inputs*tf.cast(tf.expand_dims(self.input_mask,2),dtype=tf.float32)
        
        #bilstm
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.input_lens,self.config['num_layers'],self.keep_prob)
        #layer_norm
        lstm_outputs=self.layer_norm(lstm_outputs)
        #分类
        self.logits = self.project_layer(lstm_outputs)
        #crf计算损失
        self.loss,self.trans=self.loss_layer(self.logits,self.input_lens)

        self.global_step = tf.Variable(0, trainable=False)

        trainable_params = tf.trainable_variables()
        for var in trainable_params :
            print(" trainable_params name = %s, shape = %s" % (var.name, var.shape))
        learning_rate = tf.train.exponential_decay(self.config["learning_rate"],self.global_step,
                                               self.config["decay_frequency"],self.config["decay_rate"],staircase=True)
        gradients = tf.gradients(self.loss, trainable_params)
        # apply grad clip to avoid gradient explosion
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["grad_norm"])
        # def create_optimizer(optimizer):
        #   print('create optimizer')
        #   if optimizer == "sgd":
        #       opt = tf.train.GradientDescentOptimizer(self.lr)
        #   elif optimizer == "adam":
        #       opt = tf.train.AdamOptimizer(self.lr)
        #   elif optimizer == "adgrad":
        #       opt = tf.train.AdagradOptimizer(self.lr)
        #   else:
        #       raise KeyError
        #   return opt
        # opt=create_optimizer(optimizer=self.config["optimizer"])
        opt = tf.train.AdamOptimizer(learning_rate)
        self.train_op = opt.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)
        #saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def embedding_layer(self,char_inputs):
        with tf.variable_scope('char_embedding'):
            char_lookup = tf.get_variable(name='char_embedding',shape=[self.vocab_size,self.embedding_size],initializer=self.initializer)
        embed = tf.nn.embedding_lookup(char_lookup,char_inputs)

        return embed

    def spatial_dropout(self,inputs,keep_prob):
        if keep_prob < 1:
            batch_size=shape(inputs,0)
            input_size=shape(inputs,-1)
            noise_shape = tf.stack([batch_size] + [1] + [input_size])
            inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
        return inputs

    def layer_norm(self,inputs, epsilon=1e-6):
        with tf.variable_scope("layer_norm", values=[inputs]):
            channel_size = shape(inputs,-1)
            scale = tf.get_variable("scale", shape=[channel_size],
                                    initializer=tf.ones_initializer())

            offset = tf.get_variable("offset", shape=[channel_size],
                                     initializer=tf.zeros_initializer())

            mean = tf.reduce_mean(inputs, axis=-1, keep_dims=True)
            variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1,
                                      keep_dims=True)

            norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

            return norm_inputs * scale + offset

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths,num_layers,keep_prob=1.):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        batch_size=shape(lstm_inputs,0)
        with tf.variable_scope("char_BiLSTM"):
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

    def project_layer(self, lstm_outputs):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        num_steps=shape(lstm_outputs,1)
        num_tags=len(self.label2id)
        with tf.variable_scope("project"):
            # with tf.variable_scope("hidden"):
            #     W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
            #                         dtype=tf.float32, initializer=self.initializer)

            #     b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())
            #     output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
            #     hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2,num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(output, W, b)

            return tf.reshape(pred, [-1,num_steps,num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        batch_size=shape(project_logits,0)
        num_steps=shape(project_logits,1)
        with tf.variable_scope("crf_loss"  if not name else name):
            # small = -10000.0
            # # pad logits for crf loss
            # start_logits = tf.concat(
            #     [small * tf.ones(shape=[batch_size, 1, self.num_tags]), tf.zeros(shape=[batch_size, 1, 1])], axis=-1)
            # pad_logits = tf.cast(small * tf.ones([batch_size, num_steps, 1]), tf.float32)
            # logits = tf.concat([project_logits, pad_logits], axis=-1)
            # logits = tf.concat([start_logits, logits], axis=1)
            # targets = tf.concat(
            #     [tf.cast(self.num_tags*tf.ones([batch_size, 1]), tf.int32), self.labels_ids], axis=-1)

            trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags, self.num_tags],
                initializer=self.initializer)

            log_likelihood,trans = crf_log_likelihood(
                inputs=project_logits,
                tag_indices=self.labels_ids,
                transition_params=trans,
                sequence_lengths=lengths)

            return tf.reduce_mean(-log_likelihood),trans

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        input_ids,input_mask,labels_ids,input_lens = batch
        feed_dict = {
            self.input_ids: np.asarray(input_ids),
            self.input_mask: np.asarray(input_mask),
            self.input_lens:np.asarray(input_lens)
        }
        if is_train:
            feed_dict[self.labels_ids] = np.asarray(labels_ids)

        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _= sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step,loss
        else:
            logits,lengths= sess.run([self.logits,self.input_lens],feed_dict)

            return logits,lengths

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        # small = -10000.0
        # start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            logits = score[:length]
            # pad = small * np.ones([length, 1])
            # logits = np.concatenate([score, pad], axis=-1)
            # logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path)
        return paths

    def evaluate(self, sess, data_manager, metric):
        results = []
        trans = self.trans.eval()#转移矩阵
        for batch in data_manager.iter_batch():
            input_ids, input_mask, input_tags, input_lens = batch
            scores,lengths= self.run_step(sess, False, batch)
            batch_paths = self.decode(scores,input_lens,trans)#要去padding
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=batch_paths, label_paths=target)
            
        return metric.result()

    def evaluate_line(self, sess, inputs):
      trans = self.trans.eval()
      scores,lengths= self.run_step(sess, False, inputs)
      batch_paths = self.decode(scores,lengths,trans)

      return batch_paths