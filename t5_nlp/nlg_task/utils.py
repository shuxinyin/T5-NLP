from transformers import BertTokenizer
from functools import partial

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import rouge

import jieba

jieba.setLogLevel(jieba.logging.INFO)

smooth = SmoothingFunction()
rouge = rouge.Rouge()


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def compute_bleu(label, pred, weights=None):
    '''

    '''
    weights = weights or (0.25, 0.25, 0.25, 0.25)

    return np.mean([sentence_bleu(references=[list(''.join(a))], hypothesis=list(''.join(b)),
                                  smoothing_function=smooth.method1, weights=weights)
                    for a, b in zip(label, pred)])


def compute_rouge(label, pred, weights=None, mode='weighted'):
    weights = weights or (0.2, 0.4, 0.4)
    if isinstance(label, str):
        label = [label]
    if isinstance(pred, str):
        pred = [pred]
    label = [' '.join(x) for x in label]
    pred = [' '.join(x) for x in pred]

    def _compute_rouge(label, pred):
        try:
            scores = rouge.get_scores(hyps=label, refs=pred)[0]
            scores = [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]
        except ValueError:
            scores = [0, 0, 0]
        return scores

    scores = np.mean([_compute_rouge(*x) for x in zip(label, pred)], axis=0)
    if mode == 'weighted':
        return {'rouge': sum(s * w for s, w in zip(scores, weights))}
    elif mode == '1':
        return {'rouge-1': scores[0]}
    elif mode == '2':
        return {'rouge-2': scores[1]}
    elif mode == 'l':
        return {'rouge-l': scores[2]}
    elif mode == 'all':
        return {'rouge-1': scores[0], 'rouge-2': scores[1], 'rouge-l': scores[2]}


if __name__ == '__main__':
    # BLEU: 0.3257 Rogue: {'rouge': 0.42316160588018775}
    number = 1
    hypothesis1 = [['It', 'is', 'a', 'guide', 'to', 'action', 'which',
                    'ensures', 'that', 'the', 'military', 'always',
                    'obeys', 'the', 'commands', 'of', 'the', 'party']] * number
    reference1 = [['It', 'is', 'a', 'guide', 'to', 'action', 'that',
                   'ensures', 'that', 'the', 'military', 'will', 'forever',
                   'heed', 'Party', 'commands']] * number

    hypothesis = "the #### transcript is a written version of each day 's cnn student news " \
                 "program use this transcript to help students with reading comprehension " \
                 "and vocabulary use the weekly newsquiz to test your knowledge of storie s you " \
                 "saw on cnn student news"

    reference = "this page includes the show transcript use the transcript to help students with " \
                "reading comprehension and vocabulary at the bottom of the page, " \
                "comment for a chance to be mentioned on cnn student news. " \
                "you must be a teacher or a student age # # or older to request a " \
                "mention on the cnn student news roll call . the weekly newsquiz tests " \
                "students' knowledge of even ts in the news"

    rouge = rouge.Rouge()
    scores = rouge.get_scores(hypothesis, reference)

    bleu = compute_bleu(reference1, hypothesis1)
    rouge = compute_rouge(reference1, hypothesis1, weights=None, mode='weighted')
    print(bleu)
    print(rouge)
    print([[1]] * 2)
    print(list([1, 2]))
