import os
from pathlib import Path
import logging
import random
import numpy as np
import tensorflow as tf
import pickle
import json
logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file,Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)#?

def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "best.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def create_model(session, Model_class, path,config,data,logger):
    # create model, reuse parameters if exists
    model = Model_class(config,data)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model


def json_to_text(file_path,data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


#make batch
def batchify_with_label(input_batch_list):
    # word_Ids, biword_Ids, gaz_Ids, label_Ids, gazs, gazs_count, layergazmasks
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    gazs = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    layer_gazs = [sent[4] for sent in input_batch_list]#word-->gazs
    gaz_count = [sent[5] for sent in input_batch_list]
    gaz_mask = [sent[6] for sent in input_batch_list]

    word_seq_lengths = list(map(len, words))
    max_seq_len = max(word_seq_lengths)
    word_seq_tensor = np.zeros((batch_size,max_seq_len))#0
    biword_seq_tensor= np.zeros((batch_size,max_seq_len))
    label_seq_tensor = np.zeros((batch_size,max_seq_len))#0
    mask = np.zeros((batch_size,max_seq_len))

    gaz_num = [len(layer_gazs[i][0][0]) for i in range(batch_size)]#就看第一个位置,应为其他位置已经做过padding了
    max_gaz_num = max(gaz_num)
    layer_gaz_tensor = np.zeros((batch_size, max_seq_len, 4, max_gaz_num))
    gaz_count_tensor = np.zeros((batch_size, max_seq_len, 4, max_gaz_num))
    gaz_mask_tensor = np.ones((batch_size, max_seq_len, 4, max_gaz_num))#padding是1

    for b, (seq,biseq,label, seqlen, layergaz, gazmask, gazcount,gaznum) in enumerate(zip(words,biwords,labels, word_seq_lengths, layer_gazs, gaz_mask, gaz_count,gaz_num)):

        word_seq_tensor[b, :seqlen] = np.asarray(seq)
        biword_seq_tensor[b, :seqlen]=np.asarray(biseq)
        label_seq_tensor[b, :seqlen] = np.asarray(label)
        layer_gaz_tensor[b, :seqlen, :, :gaznum] = np.asarray(layergaz)
        mask[b, :seqlen] = np.asarray([1]*int(seqlen))#padding是0
        gaz_mask_tensor[b, :seqlen, :, :gaznum] = np.asarray(gazmask)
        gaz_count_tensor[b, :seqlen, :, :gaznum] = np.asarray(gazcount)
        gaz_count_tensor[b, seqlen:] = 1#计数


    return gazs,word_seq_tensor,biword_seq_tensor,word_seq_lengths,label_seq_tensor,layer_gaz_tensor,gaz_count_tensor,gaz_mask_tensor,mask
