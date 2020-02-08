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

# config for the model
def config_model(args):
    config = OrderedDict()
    config['learning_rate']=args.learning_rate
    config["lstm_dim"]=args.hidden_size
    config['grad_norm']=args.grad_norm
    config['embedding_size']=args.embedding_size
    config['keep_prob']=0.9#train为0.9
    config['label2id']=args.label2id
    config['optimizer']="adam"
    config['num_layers']=2
    config["decay_frequency"]=100
    config["decay_rate"]=0.999
    return config


#评估,训练时调用
def evaluate(sess,args,model,processor):
    eval_dataset = load_and_cache_examples(args, processor, data_type='dev')
    eval_manager = BatchManager(data=eval_dataset, batch_size=args.batch_size,
                                 vocab=processor.vocab,label2id=args.label2id,shuffle=False)
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    model.keep_prob=1.0
    eval_info, class_info = model.evaluate(sess,eval_manager,metric)
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {}
    result = dict(result, **eval_info)
    return result, class_info


def train(args,NERModel,processor):
    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    train_manager = BatchManager(data=train_dataset, batch_size=args.batch_size,
                             vocab = processor.vocab,label2id = args.label2id,shuffle=True)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    config = config_model(args)
    config['vocab_size']=len(processor.vocab)
    loss = []
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess,NERModel, args.output_dir, config,logger)
        logger.info("start training")
        best_f1 = 0
        for epoch in range(1,1+args.epochs):
            train_manager.reset()
            
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss= model.run_step(sess, True, batch)
                loss.append(batch_loss)
            train_log = {'loss':np.mean(loss)}
            loss = []

            eval_log, class_info = evaluate(sess,args, model, processor)#!
            logs = dict(train_log, **eval_log)
            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            logger.info(show_info)
            # scheduler.epoch_step(logs['eval_f1'], epoch)
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
def predict(args,model,processor):
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    config = config_model(args)
    config['vocab_size'] = len(processor.vocab)
    config['keep_prob']=1.0
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess,NERModel,args.output_dir,config,logger)
        test_data = []
        with open(str(args.data_dir / "test.json"), 'r') as f:
            idx = 0
            for line in f:
                json_d = {}
                line = json.loads(line.strip())
                text = line['text']
                words = list(text)
                labels = ['O'] * len(words)
                json_d['id'] = idx
                json_d['context'] = " ".join(words)
                json_d['tag'] = " ".join(labels)
                json_d['raw_context'] = "".join(words)
                idx += 1
                test_data.append(json_d)
        results = []
        for step, line in enumerate(test_data):
            token_a = line['context'].split(" ")
            input_ids = [processor.vocab.to_index(w) for w in token_a]
            input_mask = [1] * len(token_a)
            input_lens = [len(token_a)]

            tags=model.evaluate_line(sess,([input_ids],[input_mask],[[]],input_lens))
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
        with open(str(args.data_dir / 'test.json'), 'r') as fr:
            for line in fr:
                test_text.append(json.loads(line))
        test_submit = []
        for x, y in zip(test_text, results):
            json_d = {}
            json_d['id'] = x['id']
            json_d['label'] = {}
            entities = y['entities']
            words = list(x['text'])
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
        examples = load_pickle(cached_examples_file)#？
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
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
    parser.add_argument("--arch", default='bilstm_crf', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=384, type=int)
    parser.add_argument("--grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--task_name", type=str, default='ner')

    args = parser.parse_args()
    args.data_dir=config.data_dir
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
    processor = CluenerProcessor(data_dir=config.data_dir)
    processor.get_vocab()
    if args.do_train:
        train(args,NERModel,processor)

    if args.do_predict:
        predict(args,NERModel,processor)


if __name__ == "__main__":
    main()
