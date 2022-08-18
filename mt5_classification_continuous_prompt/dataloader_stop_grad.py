import os
import sys

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer


def load_data(path, ind2label):
    train = pd.read_csv(path, header=0, sep='\t', names=["text", "label"])
    train = train.sample(1000, random_state=123)
    print("data shape:", train.shape)

    texts = train.text.to_list()
    labels = train.label.map(int).map(ind2label).to_list()
    # labels = train.label.map(int).to_list()
    true_labels = train.label.map(int).to_list()
    print("data head", train.head())
    return texts, labels, true_labels


class TextDataset(Dataset):
    def __init__(self, filepath, ind2label):
        super(TextDataset, self).__init__()
        self.train, self.label, self.true_label = load_data(filepath, ind2label)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        text = self.train[item]
        label = self.label[item]
        true_label = self.true_label[item]
        return text, label, true_label


class BatchTextCall(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer, max_len=100, label_len=32, use_prompt=True, num_tokens=20):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_len = label_len

        self.use_prompt = use_prompt
        # self.num_tokens = num_tokens
        # self.class_num = class_num
        # self.labels = torch.arange(num_tokens - 10, num_tokens).view(-1, 1, 1)
        # self.labels = torch.eye(class_num)

    def __call__(self, batch):

        prompt_token_text = " ".join([f"<extra_id_{ind}>" for ind in range(-13, -3)])
        # prompt_token_label = " ".join([f"<extra_id_{ind}>" for ind in range(-3, 7)])
        batch_text = [prompt_token_text + item[0] for item in batch]
        batch_label = [item[1] for item in batch]
        batch_true_label = [item[2] for item in batch]
        batch_size = len(batch_text)

        inputs = self.tokenizer(batch_text, max_length=self.max_len,
                                truncation=True, padding='max_length', return_tensors='pt')
        # 250093~250102, 250102~250112

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(batch_label, max_length=self.label_len,
                                    truncation=True, padding='max_length', return_tensors='pt')
        return inputs, labels, batch_true_label


class BatchTextCall2(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer, max_len=64, label_len=4, use_prompt=True, num_tokens=20):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # self.label_len = label_len

        self.use_prompt = use_prompt
        self.num_tokens = num_tokens
        self.labels = torch.arange(num_tokens - 10, num_tokens).view(-1, 1, 1)

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_label = [item[1] for item in batch]
        batch_true_label = [item[2] for item in batch]
        batch_size = len(batch_text)

        inputs = self.tokenizer(batch_text, max_length=self.max_len,
                                truncation=True, padding='max_length', return_tensors='pt')

        # need to pad attention_mask and input_ids to be full seq_len + n_learned_tokens
        # padding number to satisfy hf api, any input_id you padding is fine.
        # inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 1), inputs['input_ids']], 1)
        inputs['input_ids'] = torch.cat([torch.full((batch_size, self.num_tokens), 1),
                                         inputs['input_ids']], 1)
        inputs['attention_mask'] = torch.cat([torch.full((batch_size, self.num_tokens), 1),
                                              inputs['attention_mask']], 1)
        labels = self.labels[batch_label].squeeze(1)  # [batch , 1, 1]
        labels = torch.cat([torch.full((batch_size, self.num_tokens - 1), -100),
                            labels], 1)
        # decoder_input_ids = torch.full((x.size(0), self.num_tokens), 1)
        print(labels)

        return inputs, labels, batch_true_label


def test_tokenizer():
    from transformers import T5Tokenizer

    batch_text = ['下面是一则什么新闻？前门大街二期招商 部分物业拟出售',
                  '下面是一则什么新闻？融资困局得到缓解 房价与股市联动或继续推高']
    tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
    inputs = tokenizer(batch_text, max_length=64,
                       truncation=True, padding='max_length', return_tensors='pt')
    print(inputs)


if __name__ == "__main__":
    from transformers import T5Tokenizer

    print(torch.tensor([[3]]).shape)

    # print(torch.arange(0, 10).view(-1, 1, 1))
    # labels = torch.arange(0, 10).view(-1, 1, 1)
    # print(labels.shape, labels)
    # t = torch.tensor([2, 1, 0])
    # result = labels[t, :, :]
    # print(result)
    # sys.exit()

    data_dir = "../data/THUCNews/news"
    # pretrained_path = "/data/Learn_Project/Backup_Data/RoBERTa_zh_L12_PyTorch"
    pretrained_path = "/data/Learn_Project/Backup_Data/mt5-small"
    #
    label_dict = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8,
                  '财经': 9}
    ind2label_dict = dict(zip(list(label_dict.values()), list(label_dict.keys())))
    print(ind2label_dict)

    # tokenizer, model = choose_bert_type(pretrained_path, bert_type="roberta")
    tokenizer = T5Tokenizer.from_pretrained(pretrained_path)

    # test tokenizer
    # inputs = tokenizer(["测试 <extra_id_0> <extra_id_97> <extra_id_98> <extra_id_99>"], max_length=8,
    #                    truncation=True, padding='max_length', return_tensors='pt')
    # print(inputs)
    # print(tokenizer.decode(inputs.input_ids[0]))

    # model_config = BertConfig.from_pretrained(pretrained_path)
    # model = BertModel.from_pretrained(pretrained_path, config=model_config)
    text_dataset = TextDataset(os.path.join(data_dir, "test.txt"), ind2label_dict)
    text_dataset_call = BatchTextCall(tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=text_dataset_call)

    device = torch.device("cuda")
    for text, label, batch_true_label in text_dataloader:
        print(text)
        print(label)
        print(tokenizer.batch_decode(label['input_ids'][:, 28:], skip_special_tokens=True))
        print(tokenizer.batch_decode(text['input_ids'][:, 70:], skip_special_tokens=True))

        # print(tokenizer.batch_decode(list(range(250093, 250103)), skip_special_tokens=False))
        # text = text.to(device)
        break
