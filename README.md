### 一、针对自注意力更关注全局信息，捕获局部信息的能力有限，使用局部窗口限制注意力的关注范围；
### 针对底层自注意力无法捕获句子的层级语法结构，在模型底层使用on-lstm获取句子结构信息。

## 二、使用tf实现Adapting Transformer+LexiconAugment
精度有保证，超过原论文哦

## 三、RoBERTa-base+Memory network
使用RoBERTa-base+Memory network做cluener
### 1. 新建prev_trained_model文件夹，放入RoBERTa-zh-base模型
https://github.com/brightmart/roberta_zh
### 2. 新建outputs文件夹

### 3. 训练
python run_bert --do_train
### 4. 预测
python run_bert --do_predict
