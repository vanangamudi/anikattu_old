from config import Config
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import random

from .utilz import tqdm
from .debug import memory_consumed

class DataFeed(object):

    def __init__(self, name, datapoints, batchop, batch_size=1, sort_key=None):
        self.name        = name
        self._offset     = 0
        self._size       = len(datapoints)
        self._batch_size = batch_size
        self._batchop    = batchop
        self._batch_cache = {}
        self._exhausted_count = 0
        if len(datapoints):
            if sort_key:
                datapoints = sorted(datapoints, key=sort_key)
            self.bind(datapoints)

        log.info('built Datafeed: {} with the following props:'.format(self.name))
        log.info(' size       : {}'.format(self.size))
        log.info(' batch_size : {}'.format(self.batch_size))
        log.info(' num_batch  : {}'.format(self.num_batch))
        

    def bind(self, datapoints):
        self._size = len(datapoints)
        self._data = datapoints
        self._data_dict = {}

        if self.size > self.batch_size * self.num_batch:
            log.info('batch bleeds')
            self._size += self.batch_size
            
        self.reset_offset()

        for d in datapoints:
            self._data_dict[d.id] = d

    @property
    def data(self):
        return self._data

    @property
    def data_dict(self):
        return self._data_dict
    
    @property
    def size(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batch(self):
        return int(self.size/self.batch_size)
    
    @property
    def offset(self):
        return self._offset
    
    def batch(self, batch_size=None, apply_batchop=True):
        if not batch_size:
            batch_size = self.batch_size
            
        self._offset += batch_size
        b = self.data[ self.offset - batch_size   :   self.offset ]
        if apply_batchop:
            return self._batchop(b)
        return b
    
    def next_batch(self, batch_size=None, apply_batchop=True):
        try:
            if not batch_size:
                batch_size = self.batch_size
                
            if self.offset + batch_size > self.size:
                self._exhausted_count += 1
                self.reset_offset()
                
                log.debug('datafeed: {} over run - resetting offset to zero for {} time'.format(self.name, self._exhausted_count))

            return self.batch(batch_size=batch_size, apply_batchop=apply_batchop)
        except SystemExit:
            exit(1)
        except:
            log.exception('batch failed')
            return self.next_batch(apply_batchop=apply_batchop)

    def nth_batch(self, n, apply_batchop=True):
        b =    self.data[ n * self.batch_size   :   (n+1) * self.batch_size ]
        if apply_batchop:
            return self._batchop(b)
        
        return b        

    def reset_offset(self):
        self._offset = 0


class MultiplexedDataFeed(DataFeed):
    
    def __init__(self, name, datafeeds, batchop, batch_size=1, vocab=None, sort_key=None):
        self.name        = name
        self._offset     = 0
        self._size       = sum([feed.size for feed in datafeeds])
        self._batch_size = batch_size
        self._batchop    = batchop
        self.vocab = vocab
        self._batch_cache = {}
        self._exhausted_count = 0
        self.bind(datafeeds)
        log.info('built Datafeed: {} with the following props:'.format(self.name))
        log.info(' size       : {}'.format(self.size))
        log.info(' batch_size : {}'.format(self.batch_size))
        log.info(' num_batch  : {}'.format(self.num_batch))
        

    def bind(self, datafeeds):
        self.datafeeds = datafeeds
        self._data_dict = {}

        if self.size > self.batch_size * self.num_batch:
            log.info('batch bleeds')
            self._size += self.batch_size
            
        self.reset_offset()

        for datafeed in self.datafeeds:            
            for d in datafeed.data:
                self._data_dict[d.id] = d


    @property
    def data_dict(self):
        return self._data_dict
    
    @property
    def size(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batch(self):
        return int(self.size/self.batch_size)
    
    @property
    def offset(self):
        return self._offset
    
    def batch(self, batch_size=None, apply_batchop=True):
        if not batch_size:
            batch_size = self.batch_size

        b = []
        for feed in self.datafeeds:
            b.extend(feed.next_batch(batch_size=1, apply_batchop=False))
            if len(b) == batch_size: break
            
        self._offset += batch_size
        
        if apply_batchop:
            return self._batchop(b)
        return b

    '''
    def next_batch(self, batch_size=None, apply_batchop=True):
        try:
            if not batch_size:
                batch_size = self.batch_size
                
            if self.offset + batch_size > self.size:
                self.reset_offset()
            return self.batch(batch_size=batch_size, apply_batchop=apply_batchop)
        except:
            log.exception('batch failed')
            return self.next_batch(apply_batchop=apply_batchop)
    '''
    
    def nth_batch(self, n, apply_batchop=True):
        b = []
        for feed in self.datafeeds:
            b.append(random.choice(feed.nth_batch(min(n, random.choice(range(feed.num_batch))), apply_batchop=False)))
            if len(b) == self.batch_size: break
                    
        if apply_batchop:
            return self._batchop(b)
        return b


