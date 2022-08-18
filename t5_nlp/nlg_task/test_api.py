from utils import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration


def test():
    model_path = '/data/Learn_Project/Backup_Data/t5-pegasus-small'

    model = MT5ForConditionalGeneration.from_pretrained(model_path)

    tokenizer = T5PegasusTokenizer.from_pretrained(model_path)
    text = '蓝蓝的天上有一朵白白的云'
    ids = tokenizer.encode(text, return_tensors='pt')
    print(ids)
    output = model.generate(ids,
                            decoder_start_token_id=tokenizer.cls_token_id,
                            eos_token_id=tokenizer.sep_token_id,
                            max_length=30).numpy()[0]
    print(''.join(tokenizer.decode(output[:])).replace(' ', ''))
