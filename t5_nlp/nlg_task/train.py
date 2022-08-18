import os
import time
import argparse
from tqdm import tqdm
import transformers
from transformers import T5Tokenizer, MT5ForConditionalGeneration

import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import T5PegasusTokenizer, compute_bleu, compute_rouge
from dataloader import TextDataset, BatchTextCall


def evaluation(model, test_dataloader, tokenizer, device):
    # model.load_state_dict(torch.load(save_path))

    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=object)
    labels_all = np.array([], dtype=object)

    for ind, (inputs, label) in enumerate(test_dataloader):
        inputs = inputs.to(device)

        out = model.generate(inputs)
        input_decode = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        label_decode = tokenizer.batch_decode(label['input_ids'], skip_special_tokens=True)
        predict = tokenizer.batch_decode(out, skip_special_tokens=True)

        label_decode = [s.split(' ') for s in label_decode]
        predict = [s.split(' ') for s in predict]
        labels_all = np.append(labels_all, label_decode)
        predict_all = np.append(predict_all, predict)

        if ind == 0:
            print(input_decode, "\n", label_decode, '\n', predict)
    bleu_value = compute_bleu(labels_all, predict_all)
    rogue_dic = compute_rouge(labels_all, predict_all, weights=None, mode='all')
    # acc = metrics.accuracy_score(labels_all, predict_all)
    # report = metrics.classification_report(labels_all, predict_all, digits=4)
    # confusion = metrics.confusion_matrix(labels_all, predict_all)
    return bleu_value, rogue_dic


class T5_Model(nn.Module):
    """ text processed by bert model encode and get cls vector for multi classification
    refer from t5 sentiment task: https://github.com/huggingface/transformers/issues/3704
    """

    def __init__(self, model_path, pooling_type='first-last-avg'):
        super(T5_Model, self).__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, labels):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # outputs = model(input_ids=x, labels=y, attention_mask=m, decoder_input_ids=dii)

        loss = out.loss
        return loss

    def generate(self, inputs):
        out = self.model.generate(inputs.input_ids)
        # predicted_tokens = self.mt5.generate(inputs.input_ids, decoder_start_token_id=tokenizer.pad_token_id,
        #                                      num_beams=5, early_stopping=True, max_length=4)
        return out


def main(args):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # model_path = '/data/Learn_Project/Backup_Data/t5-pegasus-small'
    args.pretrained_path = "/data/Learn_Project/Backup_Data/mt5-small"
    if 'pegasus' in args.pretrained_path:
        tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrained_path)
        t5_model = T5_Model(args.pretrained_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
        t5_model = T5_Model(args.pretrained_path)

    text_dataset_call = BatchTextCall(tokenizer, max_len=64, label_len=20)

    train_text_dataset = TextDataset(os.path.join(args.data_dir, "csl_title_train.json"))
    train_dataloader = DataLoader(train_text_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, collate_fn=text_dataset_call)
    dev_text_dataset = TextDataset(os.path.join(args.data_dir, "csl_title_dev.json"))
    dev_dataloader = DataLoader(dev_text_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, collate_fn=text_dataset_call)
    test_text_dataset = TextDataset(os.path.join(args.data_dir, "csl_title_test.json"))
    test_dataloader = DataLoader(test_text_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, collate_fn=text_dataset_call)

    # t5_nlp.load_state_dict(torch.load(config.save_path))
    t5_model.to(device)
    param_optimizer = list(t5_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.lr)
    num_train_optimization_steps = len(train_text_dataset) * args.epoch
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             int(num_train_optimization_steps * args.warmup_proportion),
                                                             num_train_optimization_steps)

    loss_total = []
    for epoch in range(args.epoch):
        start_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="Training epoch{epoch}".format(epoch=epoch))

        for inputs, labels in tqdm_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            loss = t5_model(input_ids=inputs.input_ids,
                              attention_mask=inputs.attention_mask,
                              labels=labels["input_ids"])
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_total.append(loss.detach().item())
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))

        bleu_value, rogue_dic = evaluation(t5_model,
                                           dev_dataloader,
                                           tokenizer,
                                           device=device)
        print("BLEU: %.4f Rogue: %s" % (bleu_value, rogue_dic))
        time.sleep(0.1)
        # print(f"Rogue: {rogue_dic}")
        # if top_acc < acc:
        #     top_acc = acc
        #     # torch.save(multi_classification_model.state_dict(), config.save_path)
        #     print(report, '\n', confusion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T5 finetune test')
    parser.add_argument("--save_path", type=str, default="../ckpt/t5_classification")
    parser.add_argument("--data_dir", type=str, default="../../data/csl_title")
    parser.add_argument("--pretrained_path", type=str, default='/data/Learn_Project/Backup_Data/t5-pegasus-small',
                        help="pre-train model path")
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sent_max_len", type=int, default=64)
    args = parser.parse_args()

    main(args)
