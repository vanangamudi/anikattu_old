from config import Config
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn
from torch.autograd import Variable

from collections import namedtuple, defaultdict

"""
    Local Utilities, Helper Functions

"""

"""
Logging utils
"""
def logger(func, dlevel=logging.INFO):
    def wrapper(*args, **kwargs):
        level = log.getEffectiveLevel()
        log.setLevel(level)
        ret = func(*args, **kwargs)
        log.setLevel(level)
        return ret
    
    return wrapper


from pprint import pprint, pformat
from tqdm import tqdm as _tqdm
from config import Config

def tqdm(a):
    return _tqdm(a) if Config().tqdm else a


def squeeze(lol):
    """
    List of lists to List

    Args:
        lol : List of lists

    Returns:
       List 

    """
    return [ i for l in lol for i in l ]

"""
    util functions to enable pretty print on namedtuple

"""
def _namedtuple_repr_(self):
    return pformat(self.___asdict())

def ___asdict(self):
    d = self._asdict()
    for k, v in d.items():
        if hasattr(v, '_asdict'):
            d[k] = ___asdict(v)

    return dict(d)


"""
# Batching utils   
"""
import numpy as np
def seq_maxlen(seqs):
    return max([len(seq) for seq in seqs])

PAD = 0
def pad_seq(seqs, maxlen=0, PAD=PAD):
    def pad_seq_(seq):
        return seq[:maxlen] + [PAD]*(maxlen-len(seq))

    if len(seqs) == 0:
        return seqs
    
    if type(seqs[0]) == type([]):
        maxlen = maxlen if maxlen else seq_maxlen(seqs)
        seqs = [ pad_seq_(seq) for seq in seqs ]
    else:
        seqs = pad_seq_(seqs)
        
    return seqs


class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
    the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
    IPython Notebook. 
    Taken from http://calebmadrigal.com/display-list-as-table-in-ipython-notebook/"""
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

    def __repr__(self):
        lines = []
        for i in self:
            lines.append('|'.join(i))
        log.debug('number of lines: {}'.format(len(lines)))
        return '\n'.join(lines + ['\n'])

"""
torch utils
"""


def LongVar(array, requires_grad=False):
    return Var(array, requires_grad).long()

def Var(array, requires_grad=False):
    ret =  Variable(torch.Tensor(array), requires_grad=requires_grad)
    if Config.cuda:
        ret = ret.cuda()

    return ret

def init_hidden(batch_size, cell):

    layers = 1
    if not isinstance(cell, (nn.LSTMCell, nn.GRUCell)):
        layers = cell.num_layers
        if cell.bidirectional:
            layers = layers * 2

    if isinstance(cell, nn.LSTMCell):
        hidden  = Variable(torch.zeros(layers, batch_size, cell.hidden_size))
        context = Variable(torch.zeros(layers, batch_size, cell.hidden_size))
    
        if Config.cuda:
            hidden  = hidden.cuda()
            context = context.cuda()
        return hidden, context

    if isinstance(cell, nn.GRUCell):
        hidden  = Variable(torch.zeros(layers, batch_size, cell.hidden_size))
        if Config.cuda:
            hidden  = hidden.cuda()
        return hidden
    
class Averager(list):
    def __init__(self, filename=None, *args, **kwargs):
        super(Averager, self).__init__(*args, **kwargs)
        if filename:
            open(filename, 'w').close()

        self.filename = filename
        
    @property
    def avg(self):
        if len(self):
            return sum(self)/len(self)
        else:
            return 0


    def __str__(self):
        if len(self) > 0:
            return 'min/max/avg/latest: {:0.5f}/{:0.5f}/{:0.5f}/{:0.5f}'.format(min(self), max(self), self.avg, self[-1])
        
        return '<empty>'

    def append(self, a):
        try:
            super(Averager, self).append(a.data[0])
        except:
            super(Averager, self).append(a)
            
    def empty(self):
        del self[:]

    def write_to_file(self):
        if self.filename:
            with open(self.filename, 'a') as f:
                f.write(self.__str__() + '\n')
