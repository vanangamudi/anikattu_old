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

import random
import torch

from torch import optim, nn
from collections import namedtuple

from .trainer import Trainer, Predictor, Feeder, FLAGS
from ..utilz import Averager, LongVar

class Trainer(Trainer):
    def __init__(self, name, model,feeder,
                 optimizer=(None, None), 

                 loss_function = None,
                 accuracy_function = None,
                 f1score_function = None,
                 
                 initial_decoder_input = None,
                 teacher_forcing_ratio=0.5,

                 epochs=10000, checkpoint=1,

                 directory='results',
                 *args, **kwargs):
        
        self.name  = name
        assert model != (None, None)
        self.encoder_model, self.decoder_model = model
        self.__build_feeder(feeder, *args, **kwargs)

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.epochs     = epochs
        self.checkpoint = checkpoint

        self.accuracy_function = accuracy_function if accuracy_function else self._default_accuracy_function
        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.f1score_function = f1score_function

        if optimizer != (None, None):
            self.encoder_optimizer, self.decoder_optimizer = optimizer
        else:
            self.encoder_optimizer, self.decoder_optimizer = (
                optim.SGD(self.encoder_model.parameters(),lr=0.05, momentum=0.1),
                optim.SGD(self.decoder_model.parameters(),lr=0.05, momentum=0.1)
            )

        self.__build_stats(directory)
        self.best_model = (0, (self.encoder_model.state_dict(), self.decoder_model.state_dict())
        )

    def save_best_model(self):
        log.info('saving the last best model...')
        torch.save(self.best_model[1][0], '{}.{}.{}'.format(self.name, 'encoder', 'pth'))
        torch.save(self.best_model[1][1], '{}.{}.{}'.format(self.name, 'decoder', 'pth'))

    def train(self):
        self.encoder_model.train()
        self.decoder_model.train()
        for epoch in range(self.epochs):
            log.critical('memory consumed : {}'.format(memory_consumed()))         

            if self.do_every_checkpoint(epoch) == FLAGS.STOP_TRAINING:
                log.info('loss trend suggests to stop training')
                return
            
            for j in tqdm(range(self.feeder.train.num_batch)):
                log.debug('{}th batch'.format(j))
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                input_ = self.feeder.train.next_batch()
                idxs, inputs, targets = input_
                encoder_output = self.encoder_model(input_)
                loss = 0
                decoder_input = self.decoder_model.initial_input(input_, encoder_output)
                t = targets[0].transpose(0,1)
                for ti in range(t.size(0)):
                    decoder_output = self.decoder_model(input_, encoder_output, decoder_input)
                    loss_, decoder_input = self.loss_function(ti, decoder_output, input_)
                    loss += loss_
                    
                loss.backward()
                
                #nn.utils.clip_grad_norm(self.encoder_model.parameters(), Config.max_grad_norm)
                #nn.utils.clip_grad_norm(self.decoder_model.parameters(), Config.max_grad_norm)
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                self.train_loss.append(loss.data[0])
                
            log.info('-- {} -- loss: {}'.format(epoch, self.train_loss))            
            
            for m in self.metrics:
                m.write_to_file()
        
        self.encoder_model.eval()
        self.decoder_model.eval()
        return True
        
    def do_every_checkpoint(self, epoch, early_stopping=True):
        if epoch % self.checkpoint != 0:
            return

        self.encoder_model.eval()
        self.decoder_model.eval()
        for j in tqdm(range(self.feeder.test.num_batch)):
            input_ = self.feeder.test.next_batch()
            idxs, inputs, targets = input_
            encoder_output = self.encoder_model(input_)
            loss, accuracy = 0, 0
            decoder_outputs = []            
            decoder_input = self.decoder_model.initial_input(input_, encoder_output)
            t = targets[0].transpose(0,1)
            for ti in range(t.size(0)):
                decoder_output = self.decoder_model(input_, encoder_output, decoder_input)
                loss_, decoder_input = self.loss_function(ti, decoder_output, input_)
                loss += loss_
                accuracy += self.accuracy_function(ti, decoder_output, input_)
                decoder_outputs.append(decoder_input)
                        
            self.test_loss.cache(loss.data[0])
            if ti == 0: ti = 1
            self.accuracy.cache(accuracy.data[0]/ti)

            if self.f1score_function:
                precision, recall, f1score = self.f1score_function(decoder_outputs, input_)
                self.precision.append(precision)
                self.recall.append(recall)
                self.f1score.append(f1score)
                
        self.encoder_model.train()
        self.decoder_model.train()
            
        log.info('-- {} -- loss: {}, accuracy: {}'.format(epoch,
                                                          self.test_loss.epoch_cache,
                                                          self.accuracy.epoch_cache))
        if self.f1score_function:
            log.info('-- {} -- precision: {}'.format(epoch, self.precision))
            log.info('-- {} -- recall: {}'.format(epoch, self.recall))
            log.info('-- {} -- f1score: {}'.format(epoch, self.f1score))

        self.test_loss.clear_cache()
        self.accuracy.clear_cache()
        if self.best_model[0] < self.accuracy.avg:
            self.best_model = (self.accuracy.avg, (self.encoder_model.state_dict(), self.decoder_model.state_dict()))
            self.save_best_model()

        if early_stopping:
            return self.loss_trend()


            
class Predictor(object):
    def __init__(self, model, feed, repr_function, process_output, *args, **kwargs):

        self.encoder_model, self.decoder_model = model
        self.__build_feed(feed, *args, **kwargs)
        self.repr_function = repr_function
        self.process_output = process_output
        
    def __build_feed(self, feed, *args, **kwargs):
        assert feed is not None, 'feed is None, fatal error'
        self.feed = feed
        
    def predict(self,  batch_index=0, max_decoder_len=10):
        log.debug('batch_index: {}'.format(batch_index))
        idxs, i, *__ = self.feed.nth_batch(batch_index)
        self.encoder_model.eval()
        self.decoder_model.eval()
        decoder_outputs = []
        input_ = self.feed.next_batch()
        idxs, inputs, targets = input_
        encoder_output = self.encoder_model(input_)
        loss = 0

        results = ListTable()
        decoder_input = self.decoder_model.initial_input(input_, encoder_output)
        for ti in range(max_decoder_len):
            decoder_output = self.decoder_model(input_, encoder_output, decoder_input)
            decoder_output, decoder_input = self.process_output(ti, decoder_output, input_)
            decoder_outputs.append(decoder_output)
                            
        decoder_outputs = torch.stack(decoder_outputs)
        result = self.repr_function(decoder_outputs, input_)
        results.extend(result)
        return decoder_outputs, results
