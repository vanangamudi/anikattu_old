import logging
import copy
from config import CONFIG
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from ..debug import memory_consumed
from ..utilz import ListTable, Averager, tqdm
from ..utilz import are_weights_same

import torch

from torch import optim, nn
from collections import namedtuple

from nltk.corpus import stopwords

Feeder = namedtuple('Feeder', ['train', 'test'])
class FLAGS:
    CONTINUE_TRAINING = 0
    STOP_TRAINING = 1
    

class EpochAverager(Averager):
    def __init__(self, filename=None, *args, **kwargs):
        super(EpochAverager, self).__init__(filename, *args, **kwargs)
        self.epoch_cache = Averager(filename, *args, *kwargs)

    def cache(self, a):
        self.epoch_cache.append(a)

    def clear_cache(self):
        super(EpochAverager, self).append(self.epoch_cache.avg)
        self.epoch_cache.empty()
        
class Trainer(object):
    def __init__(self, name,
                 model,
                 feeder,
                 optimizer,
                 loss_function,
                 accuracy_function,
                 directory,
                 f1score_function=None,
                 epochs=10000,
                 checkpoint=1,
                 *args, **kwargs):

        self.name  = name
        self.ROOT_DIR = directory
        assert model != None
        self.model = model
        self.__build_feeder(feeder, *args, **kwargs)

        self.epochs     = epochs
        self.checkpoint = checkpoint

        self.accuracy_function = accuracy_function if accuracy_function else self._default_accuracy_function
        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.f1score_function = f1score_function
        
        self.optimizer = optimizer if optimizer else optim.SGD(self.model.parameters(),
                                                               lr=0.01, momentum=0.1)

        self.__build_stats()
        self.best_model = (0, self.model.state_dict())
        self.best_model_criteria = self.accuracy
        self.save_best_model()
        
    def __build_stats(self):
        
        # necessary metrics
        self.train_loss = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics',  'train_loss'))
        self.test_loss  = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics', 'test_loss'))
        self.accuracy   = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics', 'accuracy'))

        # optional metrics
        self.tp = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics', 'tp'))
        self.fp = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics', 'fp'))
        self.fn = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics', 'fn'))
        self.tn = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics', 'tn'))
      
        self.precision = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics', 'precision'))
        self.recall = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics', 'recall'))
        self.f1score   = EpochAverager(filename = '{}/results/{}.{}'.format(self.ROOT_DIR, 'metrics', 'f1score'))

        self.metrics = [self.train_loss, self.test_loss, self.accuracy, self.precision, self.recall, self.f1score]

        
    def __build_feeder(self, feeder, *args, **kwargs):
        assert feeder is not None, 'feeder is None, fatal error'
        self.feeder = feeder

    def save_best_model(self):
        log.info('saving the last best model with accuracy {}...'.format(self.best_model[0]))
        torch.save(self.best_model[1], '{}/weights/{:0.4f}.{}'.format(self.ROOT_DIR, self.best_model[0], 'pth'))
        torch.save(self.best_model[1], '{}/weights/{}.{}'.format(self.ROOT_DIR, self.name, 'pth'))
        
    def train(self):
        for epoch in range(self.epochs):
            log.critical('memory consumed : {}'.format(memory_consumed()))            

            if self.do_every_checkpoint(epoch) == FLAGS.STOP_TRAINING:
                log.info('loss trend suggests to stop training')
                return

            self.model.train()
            for j in tqdm(range(self.feeder.train.num_batch)):
                log.debug('{}th batch'.format(j))
                self.optimizer.zero_grad()
                input_ = self.feeder.train.next_batch()
                output = self.model(input_)
                loss = self.loss_function(output, input_)
                self.train_loss.cache(loss.data.item())

                loss.backward()
                self.optimizer.step()


            log.info('-- {} -- loss: {}'.format(epoch, self.train_loss.epoch_cache))                
            self.train_loss.clear_cache()        
                
        return True
        
    def do_every_checkpoint(self, epoch, early_stopping=True):
        if epoch % self.checkpoint != 0:
            return

        self.model.eval()
        for j in tqdm(range(self.feeder.test.num_batch)):
            input_ = self.feeder.train.next_batch()
            output = self.model(input_)
            
            loss = self.loss_function(output, input_)
            self.test_loss.cache(loss.item())
            
            accuracy = self.accuracy_function(output, input_)
            self.accuracy.cache(accuracy.item())

            if self.f1score_function:
                (tp, fn, fp, tn), precision, recall, f1score = self.f1score_function(output, input_, j)
                
                self.tp.cache(tp)
                self.fn.cache(fn)
                self.fp.cache(fp)
                self.tn.cache(tn)
                self.precision.cache(precision)
                self.recall.cache(recall)
                self.f1score.cache(f1score)

        log.info('-- {} -- loss: {}, accuracy: {}'.format(epoch, self.test_loss.epoch_cache, self.accuracy.epoch_cache))
        if self.f1score_function:
            log.info('-- {} -- tp: {}'.format(epoch, sum(self.tp.epoch_cache)))
            log.info('-- {} -- fn: {}'.format(epoch, sum(self.fn.epoch_cache)))
            log.info('-- {} -- fp: {}'.format(epoch, sum(self.fp.epoch_cache)))
            log.info('-- {} -- tn: {}'.format(epoch, sum(self.tn.epoch_cache)))
                        
            log.info('-- {} -- precision: {}'.format(epoch, self.precision.epoch_cache))
            log.info('-- {} -- recall: {}'.format(epoch, self.recall.epoch_cache))
            log.info('-- {} -- f1score: {}'.format(epoch, self.f1score.epoch_cache))

        if self.best_model[0] < self.accuracy.epoch_cache.avg:
            log.info('beat best model...')
            self.best_model = (self.accuracy.epoch_cache.avg, self.model.state_dict())
            self.save_best_model()

        self.test_loss.clear_cache()
        self.accuracy.clear_cache()
        self.tp.clear_cache()
        self.fn.clear_cache()
        self.fp.clear_cache()
        self.tn.clear_cache()
        self.f1score.clear_cache()
        self.precision.clear_cache()
        self.recall.clear_cache()
        
        for m in self.metrics:
            m.write_to_file()
            
        if early_stopping:
            return self.loss_trend()

    def loss_trend(self, total_count=10):
        if len(self.test_loss) > 4:
            losses = self.test_loss[-4:]
            count = 0
            for l, r in zip(losses, losses[1:]):
                if l < r:
                    count += 1
                    
            if count > total_count:
                return FLAGS.STOP_TRAINING

        return FLAGS.CONTINUE_TRAINING


    def _default_accuracy_function(self):
        return -1
    
class Predictor(object):
    def __init__(self, model=None,
                 feed = None,
                 repr_function = None,
                 *args, **kwargs):
        
        self.model = model
        self.__build_feed(feed, *args, **kwargs)
        self.repr_function = repr_function
                    
    def __build_feed(self, feed, *args, **kwargs):
        assert feed is not None, 'feed is None, fatal error'
        self.feed = feed
        
    def predict(self,  batch_index=0):
        log.debug('batch_index: {}'.format(batch_index))
        input_ = self.feed.nth_batch(batch_index)
        self.model.eval()
        output = self.model(input_)
        results = ListTable()
        results.extend( self.repr_function(output, input_) )
        output_ = output
        return output_, results
