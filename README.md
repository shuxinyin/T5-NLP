## T5 for nlp task:
Two nlp-task about Text Classification and Text Summary based on T5 had been tested Here.
主要基于mt5/t5-pegasus 实验了两个NLP任务，分别是基于mt5人工模板的prompt的文本分类以及生成类的文本摘要任务。

### How to run?

1. Text Classification

> cd t5_nlp/nlu_classification  
> python train.py --pretrained_path /data/Learn_Project/Backup_Data/mt5-small

2. Text Summary

- download the mt5/t5-pegasus pre-train model first.
- you can run mt5/t5-pegasus by changing the the pretrained_path here.

> cd t5_nlp/nlg_task
> python train.py --pretrained_path /data/Learn_Project/Backup_Data/t5-pegasus-small

### Result

#### Text Classification

Tested on two data sizes in training model.

|           | precision | recall | f1     | data size |
|-----------|-----------|--------|--------|-----------|
| mT5-small | 0.6434    | 0.6347 | 0.6311 | n=100     |
| mT5-small | 0.8075    | 0.7935 | 0.7954 | n=1000    |
| mT5-small | 0.8543    | 0.8546 | 0.8544 | n=10000   |

#### Text Summary

Dataset CSL: 3000 samples
Limit of my gpu memory, sentence length are set max_len=64, label_len=20, you could set it longer for getting better
result

|                  | rouge-1 | rouge-2 | rouge-l | BLEU   | config                                  |
|------------------|---------|---------|---------|--------|-----------------------------------------|
| T5-Pegasus-small | 0.5069  | 0.3030  | 0.4677  | 0.3111 | max_len=64, label_len=32, batch_size=16 |
| mT5-small        | 0.4517  | 0.3402  | 0.4251  | 0.3020 | max_len=64, label_len=20, batch_size=4  |

pretrained model you can find here:  
[T5-pegasus-torch pretrained model](https://github.com/renmada/t5-pegasus-pytorch)
[T5-pegasus-tensorflow pretrained model](https://github.com/ZhuiyiTechnology/t5-pegasus)