import math
import sys

from tqdm import tqdm
import numpy as np
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from zh_dataset_inews import title_train, label_train, title_dev, label_dev, title_test, label_test


class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        """appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])   # (batch, max_len, dim)
        # print("soft1", input_embedding.shape)
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        # print("soft2", learned_embedding.shape)  # (batch, num_prompt_tokens, dim)

        return torch.cat([learned_embedding, input_embedding], 1)


def generate_data(batch_size, n_tokens, title_data, label_data, tokenizer):
    labels = [
        torch.tensor([[3]]),  # \x00
        torch.tensor([[4]]),  # \x01
        torch.tensor([[5]]),  # \x02
    ]

    def yield_data(x_batch, y_batch, l_batch):
        x = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
        y = torch.cat(y_batch, dim=0)
        m = (x > 0).to(torch.float32)
        decoder_input_ids = torch.full((x.size(0), n_tokens), 1)
        # if torch.cuda.is_available():
        #     x = x.cuda()
        #     y = y.cuda()
        #     m = m.cuda()
        #     decoder_input_ids = decoder_input_ids.cuda()
        return x, y, m, decoder_input_ids, l_batch

    x_batch, y_batch, l_batch = [], [], []
    for x, y in zip(title_data, label_data):
        print(f"x, y: {x} {y}")
        context = x
        inputs = tokenizer(context, return_tensors="pt")
        print(inputs['input_ids'].shape)
        print(inputs)
        inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 1), inputs['input_ids']], 1)
        print(inputs['input_ids'].shape)
        print(inputs)

        l_batch.append(y)
        y = labels[y]
        y = torch.cat([torch.full((1, n_tokens - 1), -100), y], 1)
        print(y.shape, y)

        x_batch.append(inputs['input_ids'][0])
        y_batch.append(y)
        if len(x_batch) >= batch_size:
            yield yield_data(x_batch, y_batch, l_batch)
            x_batch, y_batch, l_batch = [], [], []

    if len(x_batch) > 0:
        yield yield_data(x_batch, y_batch, l_batch)
        x_batch, y_batch, l_batch = [], [], []


def main():
    mt5_pretrain = "/data/Learn_Project/Backup_Data/mt5-small"

    model = MT5ForConditionalGeneration.from_pretrained(mt5_pretrain)
    # model.cuda()
    tokenizer = T5Tokenizer.from_pretrained(mt5_pretrain)

    n_tokens = 10
    s_wte = SoftEmbedding(model.get_input_embeddings(),
                          n_tokens=n_tokens,
                          initialize_from_vocab=True)
    model.set_input_embeddings(s_wte)

    parameters = list(model.parameters())
    for x in parameters[1:]:
        x.requires_grad = False

    i = 0
    for x, y, m, dii, true_labels in generate_data(8, n_tokens, title_train, label_train, tokenizer):
        # 16 64  16-4
        print("------")
        print(x.shape, y.shape, m.shape, dii.shape)  # (batch, max_len), (batch, n_tokens)
        print(x, '\n', y, '\n', m, '\n', dii)  # (batch, max_len), (batch, n_tokens)

        assert dii.shape == y.shape
        outputs = model(input_ids=x, labels=y, attention_mask=m, decoder_input_ids=dii)
        print(outputs['logits'].shape)
        assert outputs['logits'].shape[:2] == y.shape
        pred_labels = outputs['logits'][:, -1, 3:6].argmax(-1).detach().cpu().numpy().tolist()

        i += 1
        if i == 1:
            break
    sys.exit()

    batch_size = 4
    n_epoch = 50
    total_batch = math.ceil(len(title_train) / batch_size)
    dev_total_batch = math.ceil(len(title_dev) / batch_size)
    use_ce_loss = False
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(s_wte.parameters(), lr=0.5)

    for epoch in range(n_epoch):
        print('epoch', epoch)

        all_true_labels = []
        all_pred_labels = []
        losses = []
        pbar = tqdm(enumerate(generate_data(batch_size, n_tokens, title_train, label_train, tokenizer)),
                    total=total_batch)
        for i, (x, y, m, dii, true_labels) in pbar:
            all_true_labels += true_labels

            optimizer.zero_grad()
            outputs = model(input_ids=x, labels=y, attention_mask=m, decoder_input_ids=dii)
            pred_labels = outputs['logits'][:, -1, 3:6].argmax(-1).detach().cpu().numpy().tolist()
            all_pred_labels += pred_labels

            if use_ce_loss:
                logits = outputs['logits'][:, -1, 3:6]
                # true_labels_tensor = torch.tensor(true_labels, dtype=torch.long).cuda()
                true_labels_tensor = torch.tensor(true_labels, dtype=torch.long)
                loss = ce_loss(logits, true_labels_tensor)
            else:
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().numpy().tolist()) / batch_size
            losses.append(loss_value)

            acc = accuracy_score(all_true_labels, all_pred_labels)
            pbar.set_description(f'train: loss={np.mean(losses):.4f}, acc={acc:.4f}')

        all_true_labels = []
        all_pred_labels = []
        losses = []
        with torch.no_grad():
            pbar = tqdm(enumerate(generate_data(batch_size, n_tokens, title_dev, label_dev)), total=dev_total_batch)
            for i, (x, y, m, dii, true_labels) in pbar:
                all_true_labels += true_labels
                outputs = model(input_ids=x, labels=y, attention_mask=m, decoder_input_ids=dii)
                loss = outputs.loss
                loss_value = float(loss.detach().cpu().numpy().tolist()) / batch_size
                losses.append(loss_value)
                pred_labels = outputs['logits'][:, -1, 3:6].argmax(-1).detach().cpu().numpy().tolist()
                all_pred_labels += pred_labels
                acc = accuracy_score(all_true_labels, all_pred_labels)
                pbar.set_description(f'dev: loss={np.mean(losses):.4f}, acc={acc:.4f}')


if __name__ == '__main__':
    main()
