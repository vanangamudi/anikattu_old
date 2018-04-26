import logging
from config import Config
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from ..debug import memory_consumed
from ..utilz import ListTable, Averager, tqdm

import torch

from torch import optim, nn
from collections import namedtuple

Feeder = namedtuple('Feeder', ['train', 'test'])
class FLAGS:
    CONTINUE_TRAINING = 0
    STOP_TRAINING = 1
    

class EpochAverager(Averager):
    def __init__(self, filename=None, *args, **kwargs):
        super(EpochAverager, self).__init__(*args, **kwargs)
        self.epoch_cache = Averager(filename, *args, *kwargs)

    def cache(self, a):
        self.epoch_cache.append(a)

    def clear_cache(self):
        super(EpochAverager, self).append(self.epoch_cache.avg)
        self.epoch_cache.empty()
        
class Trainer(object):
    def __init__(self, name, model=None,
                 feeder = None,
                 optimizer=None,
                 loss_function = None,
                 accuracy_function=None,
                 f1score_function=None,
                 epochs=10000, checkpoint=1,
                 directory='results',
                 *args, **kwargs):

        self.name  = name
        assert model != None
        self.model = model
        self.__build_feeder(feeder, *args, **kwargs)

        self.epochs     = epochs
        self.checkpoint = checkpoint

        self.accuracy_function = accuracy_function if accuracy_function else self._default_accuracy_function
        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.f1score_function = f1score_function
        
        self.optimizer = optimizer if optimizer else optim.SGD(self.model.parameters(),
                                                               lr=0.1, momentum=0.1)

        self.__build_stats(directory)
        self.best_model = (0, self.model.state_dict())
        
    def __build_stats(self, directory):
        
        # necessary metrics
        self.train_loss = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, self.name, 'metrics',  'train_loss'))
        self.test_loss  = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, self.name, 'metrics', 'test_loss'))
        self.accuracy   = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, self.name, 'metrics', 'accuracy'))

        # optional metrics
        self.precision = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, self.name, 'metrics', 'precision'))
        self.recall = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, self.name, 'metrics', 'recall'))
        self.f1score   = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, self.name, 'metrics', 'f1score'))

        self.metrics = [self.train_loss, self.test_loss, self.accuracy, self.precision, self.recall, self.f1score]

        
    def __build_feeder(self, feeder, *args, **kwargs):
        assert feeder is not None, 'feeder is None, fatal error'
        self.feeder = feeder

    def save_best_model(self):
        log.info('saving the last best model...')
        torch.save(self.best_model[1], '{}.{}'.format(self.name, 'pth'))
        
    def train(self, test_drive=False):
        self.model.train()
        for epoch in range(self.epochs):
            log.critical('memory consumed : {}'.format(memory_consumed()))            


            if self.do_every_checkpoint(epoch) == FLAGS.STOP_TRAINING:
                log.info('loss trend suggests to stop training')
                return

            for j in tqdm(range(self.feeder.train.num_batch)):
                log.debug('{}th batch'.format(j))
                self.optimizer.zero_grad()
                _, i, t = self.feeder.train.next_batch()
                output = self.model(*i)
                loss = self.loss_function(output, t, self.feeder.train, j)
                self.train_loss.append(loss.data[0])

                loss.backward()
                self.optimizer.step()
                
                if test_drive and j >= test_drive:
                    log.info('-- {} -- loss: {}'.format(epoch, self.train_loss))
                    return
            
            log.info('-- {} -- loss: {}'.format(epoch, self.train_loss))            
            
            
            for m in self.metrics:
                m.write_to_file()
                
        self.model.eval()
        return True
        
    def do_every_checkpoint(self, epoch, early_stopping=True):
        if epoch % self.checkpoint != 0:
            return
        self.model.eval()
        for j in tqdm(range(self.feeder.test.num_batch)):
            _, i, t = self.feeder.test.next_batch()
            output =  self.model(*i)

            loss = self.loss_function(output, t, self.feeder.test, j)
            self.test_loss.cache(loss.data[0])
            accuracy = self.accuracy_function(output, t, self.feeder.test, j)
            self.accuracy.cache(accuracy.data[0])

            if self.f1score_function:
                precision, recall, f1score = self.f1score_function(output, t, self.feeder.test, j)
                self.precision.append(precision)
                self.recall.append(recall)
                self.f1score.append(f1score)

                
        log.info('-- {} -- loss: {}, accuracy: {}'.format(epoch, self.test_loss.epoch_cache, self.accuracy.epoch_cache))
        if self.f1score_function:
            log.info('-- {} -- precision: {}'.format(epoch, self.precision))
            log.info('-- {} -- recall: {}'.format(epoch, self.recall))
            log.info('-- {} -- f1score: {}'.format(epoch, self.f1score))

        self.test_loss.clear_cache()
        self.accuracy.clear_cache()
        if early_stopping:
            return self.loss_trend()

        if self.best_model[0] < self.accuracy.avg:
            self.best_model = (self.accuracy.avg, self.model.state_dict())
            self.save_best_model()

    def loss_trend(self):
        if len(self.test_loss) > 4:
            losses = self.test_loss[-4:]
            count = 0
            for l, r in zip(losses, losses[1:]):
                if l < r:
                    count += 1
                    
            if count > 2:
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
        _, i, *__ = self.feed.nth_batch(batch_index)
        self.model.eval()
        output = self.model(*i)
        results = ListTable()
        results.extend( self.repr_function(output, self.feed, batch_index) )
        output_ = output
        return output_, results
