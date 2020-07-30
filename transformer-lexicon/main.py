from utils.data import Data
from collections import OrderedDict
import pickle
import os
import numpy as np
import config
from model import NERModel
import tensorflow as tf
import argparse
from common import init_logger,logger,seed_everything,create_model,save_model,json_to_text,batchify_with_label
from ner_metrics import SeqEntityScore
from utils_ner import get_entities
import random
import json


def config_model(args):
    config = OrderedDict()
    config['attention_size']=320
    config['ffnn_size']=2*config['attention_size']
    config['num_heads']=4
    config['attn_blocks_num']=1
    config['attention_dropout']=0.15
    config['ffnn_dropout']=0.15
    config['fc_dropout']=0.4
    config['embedding_dropout']=0.3
    config['other_learning_rate']=args.learning_rate
    config['crf_learning_rate']=0.007
    config['grad_norm']=args.grad_norm
    config['warmup_proportion']=0.01
    config['epochs']=args.epochs
    config['batch_size']=args.batch_size
    config['train_examples_len']=1350#10748
    return config

def evaluate(sess,args,model,data):
    metric = SeqEntityScore(data.label_alphabet,markup=args.markup)
    (eval_info, class_info), eval_loss = model.evaluate(sess,data.dev_Ids,metric,args.batch_size)
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss}
    result = dict(result, **eval_info)
    return result, class_info

def train(args,data,model):
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    config = config_model(args)
    with tf.Session(config=tf_config) as sess:
        best_f1 = 0
        model = create_model(sess, NERModel, args.output_dir, config,data,logger)
        logger.info("start training")
        for epoch in range(1, 1 + args.epochs):
            loss = []
            random.shuffle(data.train_Ids)
            batch_size=args.batch_size
            train_num = len(data.train_Ids)
            total_batch = train_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance = data.train_Ids[start:end]  # train_Ids
                if not instance:
                    continue
                # batchify_with_label
                #gazs, word_seq_tensor, word_seq_lengths, biword_seq_tensor, word_seq_lengths, label_seq_tensor, layer_gaz_tensor, gaz_count_tensor, gaz_mask_tensor, mask
                _,batch_word, batch_biword,batch_wordlen, batch_label, layer_gaz, gaz_count,gaz_mask,mask=batchify_with_label(instance)
                batch = (batch_word, batch_biword,batch_wordlen, batch_label, layer_gaz, gaz_count, gaz_mask,mask,True)
                step, batch_loss = model.run_step(sess,batch,True)
                # print(step)
                loss.append(batch_loss)
            train_log = {'loss': np.mean(loss)}
            loss = []
            eval_log, class_info = evaluate(sess,args,model,data)
            logs = dict(train_log, **eval_log)
            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            logger.info(show_info)
            if logs['eval_f1'] > best_f1:
                logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
                logger.info("save model to disk.")
                best_f1 = logs['eval_f1']
                save_model(sess, model, args.output_dir, logger)
                print("Eval Entity Score: ")
                for key, value in class_info.items():
                    info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                    logger.info(info)


def predict(args,data,model,mode):
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    config = config_model(args)
    if mode=='dev':
      data_Ids=data.dev_Ids
    elif mode=='test':
      data_Ids=data.test_Ids
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, NERModel, args.output_dir, config, data,logger)
        # results = []
        metric = SeqEntityScore(data.label_alphabet,markup=args.markup)
        (test_info, class_info),_= model.evaluate(sess,data_Ids,metric,batch_size=1)
        test_info = {f'test_{key}': value for key, value in test_info.items()}
        logger.info(test_info)
        for key, value in class_info.items():
          info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
          logger.info(info)
        # for step, instance in enumerate(data.test_Ids):
        #     _,batch_word, batch_wordlen,batch_label, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask=batchify_with_label([instance])
        #     batch=(batch_word, batch_wordlen,batch_label, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask,False)

            # tags = model.evaluate_line(sess,batch)
            # label_entities = get_entities(tags[0], data.label_alphabet)

        #     json_d = {}
        #     json_d['id'] = step
        #     tags[0] = [data.label_alphabet.get_instance(idx) for idx in tags[0]]
        #     json_d['tag_seq'] = " ".join(tags[0])
        #     json_d['entities'] = label_entities
        #     results.append(json_d)
        # print(" ")
        # #生成测评文件
        # output_predic_file = str(args.output_dir / "test_prediction.json")
        # output_submit_file = str(args.output_dir / "cluener_submit.json")
        # with open(output_predic_file, "w") as writer:
        #     for record in results:
        #         writer.write(json.dumps(record) + '\n')
        # test_text = []
        # with open(str(args.data_dir / 'test.json'), 'r') as fr:
        #     for line in fr:
        #         test_text.append(json.loads(line))
        # test_submit = []
        # for x, y in zip(test_text, results):
        #     json_d = {}
        #     json_d['id'] = x['id']
        #     json_d['label'] = {}
        #     entities = y['entities']
        #     words = list(x['text'])
        #     if len(entities) != 0:
        #         for subject in entities:
        #             tag = subject[0]
        #             start = subject[1]
        #             end = subject[2]
        #             word = "".join(words[start:end + 1])
        #             if tag in json_d['label']:
        #                 if word in json_d['label'][tag]:
        #                     json_d['label'][tag][word].append([start, end])
        #                 else:
        #                     json_d['label'][tag][word] = [[start, end]]
        #             else:
        #                 json_d['label'][tag] = {}
        #                 json_d['label'][tag][word] = [[start, end]]
        #     test_submit.append(json_d)
        # json_to_text(output_submit_file, test_submit)


def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_gaz_file(gaz_file)
    data.build_gaz_alphabet(train_file,count=True)
    data.build_gaz_alphabet(dev_file,count=True)
    data.build_gaz_alphabet(test_file,count=True)
    data.fix_alphabet()
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument("--do_eval", default=False, action='store_true')
    parser.add_argument('--do_predict', default=False, action='store_true')
    parser.add_argument('--markup', default='bmeso', type=str)
    parser.add_argument("--arch", default='transformer', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    # parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument("--grad_norm", default=5.0, type=float, help="Max gradient norm.")

    args = parser.parse_args()

    args.data_dir = config.data_dir
    args.output_dir = config.output_dir / '{}'.format(str(config.dataset))
    if not args.output_dir.exists():
        args.output_dir.mkdir()
    init_logger(log_file=str(args.output_dir / '{}.log'.format(args.arch)))
    seed_everything(args.seed)

    # 设置gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if os.path.exists(config.save_data_name):
        print('Loading processed data')
        with open(config.save_data_name, 'rb') as fp:
            data = pickle.load(fp)
    else:
        data=Data()
        data_initialization(data, config.gaz_file, config.train_path, config.dev_path, config.test_path)
        data.generate_instance_with_gaz(config.train_path, 'train')
        data.generate_instance_with_gaz(config.dev_path, 'dev')
        data.generate_instance_with_gaz(config.test_path, 'test')
        data.build_word_pretrain_emb(config.char_emb)
        data.build_biword_pretrain_emb(config.bichar_emb)
        data.build_gaz_pretrain_emb(config.gaz_file)
        print('Dumping data')
        with open(config.save_data_name, 'wb') as f:
            pickle.dump(data, f)
    if args.do_train:
        train(args,data,NERModel)
    
    if args.do_predict:
        predict(args,data,NERModel,'dev')

if __name__=='__main__':
    main()