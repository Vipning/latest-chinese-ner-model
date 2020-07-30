from pathlib import Path

dataset = Path('weibo')
data_dir = Path("./data")/dataset
train_path = data_dir / 'train.json'
dev_path =data_dir / 'dev.json'
test_path = data_dir / 'test.json'
output_dir = Path("./outputs")
char_emb = "data/gigaword_chn.all.a2b.uni.ite50.vec"
bichar_emb = "data/gigaword_chn.all.a2b.bi.ite50.vec"
gaz_file = "data/ctb.50d.vec"
save_data_name=data_dir/'data.dset'