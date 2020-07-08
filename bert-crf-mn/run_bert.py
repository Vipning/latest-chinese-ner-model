# encoding=utf8
from collections import OrderedDict
import tensorflow as tf
import argparse
import config
from common import init_logger,logger,seed_everything,load_pickle,save_pickle,create_model,save_model,json_to_text
import os
from data_processor import CluenerProcessor
from dataset_loader import BatchManager
from model import NERModel
import numpy as np
from ner_metrics import SeqEntityScore
from utils_ner import get_entities
import json
import tokenization
# config for the model
def config_model(args):
    config = OrderedDict()
    # config["lstm_dim"]=512
    # config['lstm_dropout']=0.1
    # config['lstm_layers']=1
    config['key_size']=512
    config['value_size']=512
    config['sa_output_size']=512
    config['num_heads']=8
    config['attention_dropout']=0.1
    config['dropout_rate']=0.1#train
    config['label2id']=args.label2id
    config['warmup_proportion']=0.1
    config["epochs"]=args.epochs
    config['train_examples_len']=10748
    config['batch_size']=args.batch_size
    config['init_checkpoint']=args.init_checkpoint
    config['tf_checkpoint']=args.tf_checkpoint
    config['max_seq_len']=args.max_seq_len
    config['bert_config_file']=args.bert_config_file
    config['label_embedding_size'] = 304
    config['aug_num'] = args.aug_num
    config['label_embedding'] = args.label_embedding
    config['bert_learning_rate']=3e-5
    config['task_learning_rate']=2e-4
    config['adam_eps']=1e-6
    config['task_optimizer']='adam'
    return config


#评估,训练时调用
def evaluate(sess,args,model,processor):
    eval_dataset = load_and_cache_examples(args, processor, data_type='dev')
    eval_manager = BatchManager(data=eval_dataset, batch_size=args.batch_size,shuffle=False)
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    (eval_info,class_info),eval_loss= model.evaluate(sess,eval_manager,metric)
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss':eval_loss}
    result = dict(result, **eval_info)
    return result, class_info


def train(args,NERModel,processor):

    train_dataset = load_and_cache_examples(args,processor,data_type='train')
    train_manager = BatchManager(data=train_dataset,batch_size=args.batch_size,shuffle=True)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    config = config_model(args)
    loss = []
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess,NERModel, args.output_dir, config,logger)
        logger.info("start training")
        best_f1 = 0
        for epoch in range(1,1+args.epochs):
            train_manager.reset()
            for batch in train_manager.iter_batch(shuffle=True):
                instance = (batch, True)
                step, batch_loss= model.run_step(sess,instance,True)
                loss.append(batch_loss)
            train_log = {'loss':np.mean(loss)}
            loss = []

            eval_log, class_info = evaluate(sess,args, model, processor)
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

#测试
def predict(args,processor):
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    config = config_model(args)
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess,NERModel,args.output_dir,config,logger)
        test_data = []
        with open(str(args.data_dir / "test.json"), 'r') as f:
            idx = 0
            for line in f:
                tokens=[]
                json_d = {}
                line = json.loads(line.strip())
                textlist = list(line['text'])
                for i, word in enumerate(textlist):
                    token = tokenizer.tokenize(word)
                    assert len(token)==1
                    tokens.extend(token)
                assert len(tokens) < args.max_seq_len
                ntokens = []
                segment_ids = []
                label_ids = []
                ntokens.append("[CLS]")  # 句子开始设置CLS 标志
                segment_ids.append(0)
                for i, token in enumerate(tokens):
                    ntokens.append(token)
                    segment_ids.append(0)
                ntokens.append("[SEP]")
                segment_ids.append(0)
                # append("O") or append("[SEP]") not sure!
                input_ids = tokenizer.convert_tokens_to_ids(ntokens)
                input_len = len(input_ids)
                input_mask = [1] * len(input_ids)
                while len(input_ids) < args.max_seq_len:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                raw_text=[]
                raw_text.append('[CLS]')
                raw_text.extend(textlist)
                raw_text.append('[SEP]')
                assert len(raw_text) ==len(ntokens)
                assert len(input_ids) == args.max_seq_len
                assert len(input_mask) == args.max_seq_len
                assert len(segment_ids) == args.max_seq_len
                
                json_d['id']=idx
                json_d['input_ids'] = input_ids
                json_d['input_mask'] = input_mask
                json_d['segment_ids'] = segment_ids
                json_d['input_len'] = input_len
                json_d['text'] = raw_text
                idx += 1
                test_data.append(json_d)
        results = []
        train_data = processor.get_train_examples()
        test_train = load_pickle(args.data_dir / 'train_test.bin')
        for step, line in enumerate(test_data):
            a_input_ids = []
            a_input_mask = []
            a_label_ids = []
            a_input_lens = []
            a_segment_ids=[]
            aux_sentence = [train_data[i] for i in test_train[step][:args.aug_num]]
            for s in aux_sentence:
                a_input_ids.append(s['input_ids'])
                a_label_ids.append(s['label_ids'])
                #地址信息增强，将所有的标签信息改成adress标签，全1
                #a_label_ids.append(s['input_mask'])
                a_input_mask.append(s['input_mask'])
                a_input_lens.append(s['input_len'])
                a_segment_ids.append(s['segment_ids'])
            input_ids = line['input_ids']
            input_mask = line['input_mask']
            input_lens = line['input_len']
            segment_ids = line['segment_ids']
            batch = {
                'ori': ([input_ids], [input_mask], [[]], [input_lens],[segment_ids]),
                'aug': ([a_input_ids], [a_input_mask], [a_label_ids], [a_input_lens],[a_segment_ids])
            }
            tags = model.evaluate_line(sess,batch)
            label_entities = get_entities(tags[0], args.id2label)
            json_d = {}
            json_d['id'] = step
            tags[0] = [args.id2label[idx] for idx in tags[0]]
            json_d['tag_seq'] = " ".join(tags[0])
            json_d['entities'] = label_entities
            results.append(json_d)
        print(" ")
        output_predic_file = str(args.output_dir / "test_prediction.json")
        output_submit_file = str(args.output_dir / "cluener_submit.json")
        with open(output_predic_file, "w") as writer:
            for record in results:
                writer.write(json.dumps(record) + '\n')
        test_text = []
        
        test_submit = []
        for x, y in zip(test_data,results):
            json_d = {}
            json_d['id'] = x['id']
            json_d['label'] = {}
            entities = y['entities']
            #加了标记
            words = x['text']
            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d['label']:
                        if word in json_d['label'][tag]:
                            json_d['label'][tag][word].append([start, end])
                        else:
                            json_d['label'][tag][word] = [[start, end]]
                    else:
                        json_d['label'][tag] = {}
                        json_d['label'][tag][word] = [[start, end]]
            test_submit.append(json_d)
        json_to_text(output_submit_file, test_submit)

def load_and_cache_examples(args,processor, data_type='train'):
    # Load data features from cache or dataset file
    cached_examples_file = args.data_dir / 'cached_crf-{}_{}_{}'.format(
        data_type,
        args.arch,#结构
        str(args.task_name))
    if cached_examples_file.exists():
        logger.info("Loading features from cached file %s", cached_examples_file)
        examples = load_pickle(cached_examples_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == 'train':
            examples = processor.get_aug_examples(args.data_dir/'train_train.bin',args.aug_num,data_type)
        elif data_type == 'dev':
            examples = processor.get_aug_examples(args.data_dir/'train_dev.bin',args.aug_num,data_type)
        logger.info("Saving features into cached file %s", cached_examples_file)
        save_pickle(examples, str(cached_examples_file))
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train",default=False,action='store_true')
    parser.add_argument("--do_eval",default=False,action='store_true')
    parser.add_argument('--do_predict',default=False,action='store_true')
    #bios,s是single单字
    parser.add_argument('--markup', default='bios', type=str, choices=['bio', 'bios'])
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument("--task_name", type=str, default='ner')
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--aug_num", default=4, type=int)
    
    args = parser.parse_args()
    args.vocab_file=config.vocab_file
    args.data_dir=config.data_dir
    args.tf_checkpoint=config.tf_checkpoint
    args.init_checkpoint=config.init_checkpoint
    args.bert_config_file=config.bert_config_file
    
    if not config.output_dir.exists():
        args.output_dir.mkdir()
    args.output_dir = config.output_dir / '{}'.format(args.arch)
    if not args.output_dir.exists():
        args.output_dir.mkdir()
    init_logger(log_file=str(args.output_dir / '{}-{}.log'.format(args.arch, args.task_name)))
    seed_everything(args.seed)
    #设置gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.id2label = {i: label for i, label in enumerate(config.label2id)}
    args.label2id = config.label2id

    processor = CluenerProcessor(data_dir=config.data_dir,vocab_file=config.vocab_file,label2id=config.label2id,max_seq_len=args.max_seq_len)
    processor.get_label_embedding(args.label2id, config.pretrain_label_embedding_file, config.label_embedding_file)
    args.label_embedding = processor.label_embedding
    if args.do_train:
        train(args,NERModel,processor)

    if args.do_predict:
        predict(args,processor)


if __name__ == "__main__":
    main()
