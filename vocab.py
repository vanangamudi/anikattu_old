
import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Vocab:

    def __init__(self, vocab, special_tokens, sort_key=None):

        log.info('Constructiong vocabuluary object...')
        self.vocab = vocab

        index2word = list(vocab.keys())
        if sort_key:
            index2word = sorted(index2word, key=sort_key)

        self.index2word = special_tokens + index2word        
        self.word2index = {w:i for i, w in enumerate(self.index2word)}

        log.info('number of word in index2word and word2index: {} and {}'
                 .format(len(self.index2word), len(self.word2index)))
        
    def __getitem__(self, key):
        if type(key) == int:
            return self.index2word[key]
        elif type(key) == str:
            return self.word2index[key]


    def __len__(self):
        return len(self.index2word)
