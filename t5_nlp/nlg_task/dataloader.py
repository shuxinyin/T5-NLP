import os
import json

from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, json_file):
        super(TextDataset, self).__init__()
        self.json_data = self.load_json(json_file)
        print(f"data size: {len(self.json_data)}")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, item):
        json_element = self.json_data[item]

        return json_element

    def load_json(self, path):
        return [json.loads(x) for x in open(path, encoding='utf-8')]


class BatchTextCall(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer, max_len=64, label_len=32):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_len = label_len

    def __call__(self, batch):
        batch_text = [item['abst'] for item in batch]
        batch_label = [item['title'] for item in batch]

        inputs = self.tokenizer(batch_text, max_length=self.max_len,
                                truncation=True, padding='max_length', return_tensors='pt')
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(batch_label, max_length=self.label_len,
                                    truncation=True, padding='max_length', return_tensors='pt')

        return inputs, labels


if __name__ == "__main__":
    import argparse
    from transformers import T5Tokenizer, MT5ForConditionalGeneration
    from utils import T5PegasusTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default="../../data/csl_title")
    parser.add_argument('--dev_file', type=str, )
    parser.add_argument('--predict_file', type=str, )
    args = parser.parse_args()

    data_dir = "../data/THUCNews/news"
    model_path = '/data/Learn_Project/Backup_Data/t5-pegasus-small'
    tokenizer = T5PegasusTokenizer.from_pretrained(model_path)

    data_dir = "../../data/csl_title"

    text_dataset = TextDataset(os.path.join(data_dir, "csl_title_dev.json"))
    text_dataset_call = BatchTextCall(tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=2, shuffle=True,
                                 num_workers=2, collate_fn=text_dataset_call)
    for text, label in text_dataloader:
        print(text)
        print(label)
        print(tokenizer.decode(text.input_ids[0]))
        print(tokenizer.decode(label.input_ids[0]))
        # text = text.to(device)
        # print(out)
        # predict = tokenizer.batch_decode(out, skip_special_tokens=True)
        # print(predict)
