## 一、使用tf实现Adapting Transformer+LexiconAugment
精度有保证，超过原论文哦
## 二、RoBERTa-base+Memory network
使用RoBERTa-base+Memory network做cluener
### 1. 新建prev_trained_model文件夹，放入RoBERTa-zh-base模型
https://github.com/brightmart/roberta_zh
### 2. 新建outputs文件夹

### 3. 训练
python run_bert --do_train
### 4. 预测
python run_bert --do_predict
