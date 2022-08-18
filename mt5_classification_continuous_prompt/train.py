import os
import re
import sys
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer
# from transformers import MT5Model, T5Tokenizer
from transformers import MT5ForConditionalGeneration, T5Tokenizer

# from dataloader import TextDataset, BatchTextCall
from dataloader_stop_grad import TextDataset, BatchTextCall
from model import MultiClassT5, SoftPromptEmbedding

import warnings

warnings.filterwarnings('ignore')


def choose_bert_type(path, bert_type="tiny_albert"):
    """
    choose bert type for chinese, tiny_albert or macbert（bert）
    return: tokenizer, model
    """

    if bert_type == "albert":
        model_config = BertConfig.from_pretrained(path)
        model = AlbertModel.from_pretrained(path, config=model_config)
    elif bert_type == "bert" or bert_type == "roberta":
        model_config = BertConfig.from_pretrained(path)
        model = BertModel.from_pretrained(path, config=model_config)
    elif bert_type == 't5':
        model_config = BertConfig.from_pretrained(path)
        model = MT5ForConditionalGeneration.from_pretrained(path)
    else:
        model_config, model = None, None
        print("ERROR, not choose model!")

    return model_config, model


def evaluation(model, test_dataloader, tokenizer, label2ind_dict, device, num_tokens=20):
    # model.load_state_dict(torch.load(save_path))

    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for ind, (inputs, labels, true_labels) in enumerate(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        out = model.generate(inputs)
        label_decode = tokenizer.batch_decode(labels['input_ids'], skip_special_tokens=True)
        # print(tokenizer.batch_decode(labels['input_ids'][:, 28:], skip_special_tokens=True))

        predict = tokenizer.batch_decode(out, skip_special_tokens=True)

        # outputs = model(inputs, labels["input_ids"], use_prompt=True)
        # # print(outputs['logits'].shape)  # (batch, nums_tokens, V=250112)
        # logits = outputs['logits'][:, -1, num_tokens - 10:num_tokens]
        # print(outputs['logits'].shape)  # (batch, nums_tokens, V=250112)

        # true_labels_tensor = torch.tensor(true_labels, dtype=torch.long).to(device)
        # pred_labels = logits.argmax(-1).detach().cpu().numpy().tolist()
        # print(pred_labels)  # (batch, nums_tokens, V=250112)

        labels_all = np.append(labels_all, label_decode)
        predict_all = np.append(predict_all, predict)

        # print(label_decode, '\n', predict)
        # print(label_decode_filter, '\n', predict_filter)
        # print(len(label_decode), len(predict))
        # print(labels_all.shape, predict_all.shape)
    print(labels_all, '\n', predict_all)
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return acc, report, total_loss / len(test_dataloader), confusion


def train(config):
    label2ind_dict = {'finance': 0, 'realty': 1, 'stocks': 2, 'education': 3, 'science': 4, 'society': 5, 'politics': 6,
                      'sports': 7, 'game': 8, 'entertainment': 9}

    ind2label_dict = dict(zip(list(label2ind_dict.values()), list(label2ind_dict.keys())))

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    mt5_pretrain = "/data/Learn_Project/Backup_Data/mt5-small"
    model_t5 = MT5ForConditionalGeneration.from_pretrained(mt5_pretrain)
    tokenizer = T5Tokenizer.from_pretrained(mt5_pretrain)

    # num_tokens = 20
    stop_grad_tokens_count = 250093
    # train_dataset_call = BatchTextCall(tokenizer, max_len=config.sent_max_len, use_prompt=True, num_tokens=num_tokens)

    train_dataset_call = BatchTextCall(tokenizer, use_prompt=True)
    train_dataset = TextDataset(os.path.join(config.data_dir, "train.txt"), ind2label_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True, num_workers=6,
                                  collate_fn=train_dataset_call)

    test_dataset = TextDataset(os.path.join(config.data_dir, "test.txt"), ind2label_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=True, num_workers=6,
                                 collate_fn=train_dataset_call)

    # print(f"use prompt embedding")
    # soft_embedding = SoftPromptEmbedding(model_t5.get_input_embeddings(),
    #                                      num_tokens=num_tokens,
    #                                      initialize_from_vocab=False)

    # use prompt embedding replace the original embedding layer
    # model_t5.set_input_embeddings(soft_embedding)
    parameters = list(model_t5.parameters())
    for x in parameters[1:]:
        x.requires_grad = False

    multi_classification_model = MultiClassT5(model_t5, pooling_type=config.pooling_type)
    optimizer = transformers.AdamW(multi_classification_model.parameters(), lr=config.lr)

    # multi_classification_model.load_state_dict(torch.load(config.save_path))
    multi_classification_model.to(device)

    num_train_optimization_steps = len(train_dataloader) * config.epoch

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             int(num_train_optimization_steps * config.warmup_proportion),
                                                             num_train_optimization_steps)
    ce_loss = F.cross_entropy

    loss_total, top_acc = [], 0
    for epoch in range(config.epoch):
        # multi_classification_model.train()
        start_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="Training epoch{epoch}".format(epoch=epoch))
        for i, (inputs, labels, true_labels) in enumerate(tqdm_bar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs["input_ids"].shape, labels.shape)

            # print(loss)
            if not config.use_prompt:
                outputs = multi_classification_model(inputs, labels, use_prompt=config.use_prompt)
                # print("outputs logits shape", outputs['logits'].shape)  # (batch, nums_tokens, V=250112)

                # logits = outputs['logits'][:, -1, num_tokens - 10:num_tokens]
                # print("logits shape", logits.shape)
                # print("logits", logits)
                true_labels_tensor = torch.tensor(true_labels, dtype=torch.long).to(device)
                # print("true_labels_tensor shape", true_labels_tensor.shape)

                # loss = ce_loss(logits, true_labels_tensor)
            else:
                out = multi_classification_model(inputs, labels["input_ids"])
                loss = out.loss
            loss.backward()
            indices = torch.LongTensor(list(range(stop_grad_tokens_count)))
            model_t5.shared.weight.grad[indices] = 0
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_total.append(loss.detach().item())
        logger.info("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))

        acc, report, loss, confusion = evaluation(multi_classification_model,
                                                  test_dataloader, tokenizer,
                                                  label2ind_dict, device=device)
        logger.info("Accuracy: %.4f Loss in test %.4f" % (acc, loss))
        if top_acc < acc:
            top_acc = acc
            # torch.save(multi_classification_model.state_dict(), config.save_path)
            print(report, '\n', confusion)
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T5 finetune test')
    parser.add_argument("--data_dir", type=str, default="../data/THUCNews/news")
    parser.add_argument("--save_path", type=str, default="../ckpt/t5_classification")
    parser.add_argument("--pretrained_path", type=str, default="/data/Learn_Project/Backup_Data/mt5-small",
                        help="pre-train model path")
    parser.add_argument("--bert_type", type=str, default="bert", help="bert or albert")
    parser.add_argument("--use_prompt", type=bool, default=True, help="use prompt or not")
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--epoch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--pooling_type", type=str, default="first-last-avg")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sent_max_len", type=int, default=54)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d:%(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    log_filename = f"t5_classification.log"
    logger.addHandler(logging.FileHandler(os.path.join("./log", log_filename), 'w'))
    logger.info(args)
    print('use_prompt', bool(args.use_prompt))
    train(args)
