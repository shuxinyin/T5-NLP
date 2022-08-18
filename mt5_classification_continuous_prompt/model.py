import sys

import numpy.random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

import torch
from torch import nn
from transformers import MT5Model, T5Tokenizer, MT5ForConditionalGeneration

from torch.nn import CrossEntropyLoss


class MultiClassT5(nn.Module):
    """ text processed by bert model encode and get cls vector for multi classification
    refer from t5 sentiment task: https://github.com/huggingface/transformers/issues/3704
    """

    def __init__(self, mt5_model, pooling_type='first-last-avg'):
        super(MultiClassT5, self).__init__()
        self.mt5 = mt5_model
        self.pooling = pooling_type
        # self.linear = nn.Linear(250112, 10)

    def forward(self, inputs, labels, use_prompt=False):
        out = self.mt5(**inputs, labels=labels)

        # if use_prompt:
        #     out_model = out.logits
        # else:
        #     out_model = out.loss

        # loss = out.loss
        # loss.require_grad = True

        return out

    def generate(self, inputs):
        # print('--', inputs.input_ids.shape)
        # print(inputs.input_ids[:, 20:].shape)
        out = self.mt5.generate(inputs.input_ids)
        # predicted_tokens = self.mt5.generate(inputs.input_ids, decoder_start_token_id=tokenizer.pad_token_id,
        #                                      num_beams=5, early_stopping=True, max_length=4)
        return out


class SoftPromptEmbedding(nn.Module):
    def __init__(self, t5_token_embedding: nn.Embedding, num_tokens=20, initialize_from_vocab=True):
        """ This class is used to attach a for learning to the model embedding
        refer: https://github.com/kipgparker/soft-prompt-tuning
        Args:
            t5_token_embedding (nn.Embedding):  embedding of t5 model tokens.
            num_tokens (int, optional): number of tokens for task.
            init_from_vocab (bool, optional): initialed from default vocab. Defaults to True.
        """
        super(SoftPromptEmbedding, self).__init__()
        self.t5_token_embedding = t5_token_embedding
        self.num_tokens = num_tokens

        if initialize_from_vocab:
            params = self.t5_token_embedding.weight[:num_tokens].clone().detach()
        else:
            params = nn.init.xavier_normal_(torch.empty(num_tokens, t5_token_embedding.weight.size()[1]))
            # params = torch.FloatTensor(num_tokens, t5_token_embedding.weight.size()[1]).uniform_(-0.5, 0.5)
        self.prompt_embedding = nn.parameter.Parameter(params)
        print("prompt_embedding shape", self.prompt_embedding.shape)

    def forward(self, tokens):
        """
        Args: tokens (torch.long): input tokens before encoding
        Returns: torch.float: encoding of text concatenated with learned task specific embedding
        """
        # print('tokens:', tokens)
        input_embedding = self.t5_token_embedding(tokens[:, self.num_tokens:])  # (batch, max_len-num_tokens, dim)
        prompt_embedding = self.prompt_embedding.repeat(input_embedding.size(0), 1, 1)  # (batch, num_tokens, dim)
        # prompt_embedding = self.prompt_embedding(tokens[:, :self.num_tokens])  # (batch, num_tokens, dim)
        # print("embedding shape", input_embedding.shape, prompt_embedding.shape)

        # actual embedding = [task prompt embedding, input_embedding]
        return torch.cat([prompt_embedding, input_embedding], 1)


def test_prompt_embedding():
    mt5_pretrain = "/data/Learn_Project/Backup_Data/mt5-small"
    model_t5 = MT5ForConditionalGeneration.from_pretrained(mt5_pretrain)
    tokenizer = T5Tokenizer.from_pretrained(mt5_pretrain)
    print(tokenizer.pad_token_id)
    print(tokenizer.batch_decode([250093, 250099, 250100, 250112], skip_special_tokens=True))
    # ['<extra_id_6>', '<extra_id_0>', '<extra_id_-1>', '<extra_id_-13>']

    stop_grad_tokens_count = 250093
    # num_tokens = 10
    # t5_model_embedding = model_t5.get_input_embeddings()
    # print(t5_model_embedding.weight.size())
    # soft_embedding = SoftPromptEmbedding(model_t5.get_input_embeddings(),
    #                                      num_tokens=num_tokens,
    #                                      initialize_from_vocab=True)

    # make prompt embedding replace the original embedding layer
    # model_t5.set_input_embeddings(soft_embedding)

    parameters = list(model_t5.parameters())
    for x in parameters[1:]:
        x.requires_grad = False

    article = ["UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."] * 2
    summary = ["Weiter Verhandlung in Syrien."] * 2

    inputs = tokenizer(article, max_length=25, truncation=True, padding='max_length', return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summary, max_length=16, truncation=True, padding='max_length', return_tensors='pt')
    print(inputs.input_ids)

    # outputs = model(inputs, labels)
    outputs = model_t5(**inputs, labels=labels["input_ids"])
    print(outputs.logits.shape)
    loss = outputs.loss
    loss.backward()
    # param_optimizer = list(model_t5.named_parameters())
    # print(param_optimizer[:2])
    indices = torch.LongTensor(list(range(stop_grad_tokens_count)))
    model_t5.shared.weight.grad[indices] = 0
    print("--", model_t5.shared.weight)
    print(model_t5.shared.weight.grad[indices])
    # print(model_t5.shared.t5_token_embedding.weight.grad)

    outputs = model_t5.generate(**inputs)
    print(outputs)


def test_batch_train():
    import os
    from dataloader import BatchTextCall, TextDataset
    from torch.utils.data import DataLoader
    mt5_pretrain = "/data/Learn_Project/Backup_Data/mt5-small"
    model_t5 = MT5ForConditionalGeneration.from_pretrained(mt5_pretrain)
    tokenizer = T5Tokenizer.from_pretrained(mt5_pretrain)

    label2ind_dict = {'finance': 0, 'realty': 1, 'stocks': 2, 'education': 3, 'science': 4, 'society': 5,
                      'politics': 6, 'sports': 7, 'game': 8, 'entertainment': 9}
    ind2label_dict = dict(zip(list(label2ind_dict.values()), list(label2ind_dict.keys())))

    multi_classification_model = MultiClassT5(model_t5)
    train_dataset_call = BatchTextCall(tokenizer, max_len=64)

    train_dataset = TextDataset(os.path.join("../data/THUCNews/news", "train.txt"), ind2label_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6,
                                  collate_fn=train_dataset_call)

    for i, (inputs, labels, true_labels) in enumerate(train_dataloader):
        print(inputs["input_ids"].shape)

        # outputs = multi_classification_model(inputs, labels)
        predict = multi_classification_model.generate(inputs)

        # print(outputs)
        print(predict)
        print(tokenizer.batch_decode(predict, skip_special_tokens=True))

        sys.exit()


if __name__ == '__main__':
    test_prompt_embedding()
    # test_batch_train()