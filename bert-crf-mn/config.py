from pathlib import Path
data_dir = Path("./data")
train_path = data_dir / 'train.json'
dev_path =data_dir / 'dev.json'
test_path = data_dir / 'test.json'
output_dir = Path("./outputs")
vocab_file='prev_trained_model/vocab.txt'
bert_config_file='prev_trained_model/bert_config.json'
tf_checkpoint='prev_trained_model/bert_model.ckpt'
init_checkpoint='prev_trained_model/bert_model.ckpt'
pretrain_label_embedding_file=data_dir / 'pretrain_label_embedding.txt'
label_embedding_file=data_dir / 'label_embedding.npy'

label2id = {
    "O": 0,
    "B-address":1,
    "B-book":2,
    "B-company":3,
    'B-game':4,
    'B-government':5,
    'B-movie':6,
    'B-name':7,
    'B-organization':8,
    'B-position':9,
    'B-scene':10,
    "I-address":11,
    "I-book":12,
    "I-company":13,
    'I-game':14,
    'I-government':15,
    'I-movie':16,
    'I-name':17,
    'I-organization':18,
    'I-position':19,
    'I-scene':20,
    "S-address":21,
    "S-book":22,
    "S-company":23,
    'S-game':24,
    'S-government':25,
    'S-movie':26,
    'S-name':27,
    'S-organization':28,
    'S-position':29,
    'S-scene':30
}
