import os
import sys

sys.path.append('.')
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
from ..utilz import ListTable, Averager, tqdm, init_hidden
from ..utilz import are_weights_same


from .trainer import EpochAverager, FLAGS
from .trainer import Trainer, Tester
from ..utilz import Var, LongVar


import torch

from torch import optim, nn
from collections import namedtuple

from nltk.corpus import stopwords

class Trainer(Trainer):
    def __init__(self, name,
                 config,
                 model,
                 
                 feed,
                 optimizer,
                 
                 loss_function,
                 directory,
                 
                 teacher_forcing_ratio=0.5,                 
                 
                 epochs=1000,
                 checkpoint=1,
                 do_every_checkpoint=None,
                 
                 *args,
                 **kwargs
                 
    ):

        self.name  = name
        self.config = config
        self.ROOT_DIR = directory

        self.log = logging.getLogger('{}.{}.{}'.format(__name__, self.__class__.__name__, self.name))

        self.model = model
        self.feed = feed

        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        self.epochs     = epochs
        self.checkpoint = min(checkpoint, epochs)

        self.do_every_checkpoint = do_every_checkpoint if not do_every_checkpoint == None else lambda x: FLAGS.CONTINUE_TRAINING

        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.optimizer = optimizer if optimizer else optim.SGD(self.model.parameters(),
                                                               lr=0.01, momentum=0.1)

        self.__build_stats()
        self.best_model = (0, self.model.state_dict())
                           
        if self.config.CONFIG.cuda:
            self.model.cuda()

    def train(self):
        for epoch in range(self.epochs):
            self.log.critical('memory consumed : {}'.format(memory_consumed()))            

            if epoch % max(1, (self.checkpoint - 1)) == 0:
                if self.do_every_checkpoint(epoch) == FLAGS.STOP_TRAINING:
                    self.log.info('loss trend suggests to stop training')
                    return
                           
            self.model.train()
            for j in tqdm(range(self.feed.num_batch), desc='Trainer.{}'.format(self.name)):
                input_ = self.feed.next_batch()
                idxs, inputs, targets = input_
                sequence = inputs[0].transpose(0,1)
                _, batch_size = sequence.size()

                state = self.model.initial_hidden(batch_size)
                loss = 0
                output = sequence[0]
                for ti in range(1, sequence.size(0) - 1):
                    output = self.model(output, state)
                    loss += self.loss_function(ti, output, input_)
                    output, state = output
                    output = output.max(1)[1]
                    
                loss.backward()
                self.train_loss.cache(loss.data.item())
                #nn.utils.clip_grad_norm(self.encoder_model.parameters(), Config.max_grad_norm)
                #nn.utils.clip_grad_norm(self.decoder_model.parameters(), Config.max_grad_norm)
                self.optimizer.step()


            self.log.info('-- {} -- loss: {}\n'.format(epoch, self.train_loss.epoch_cache))
            self.train_loss.clear_cache()
            
            for m in self.metrics:
                m.write_to_file()

        return True


class Tester(Tester):
    def __init__(self, name,
                 config,
                 model,
                 feed,
                 loss_function,
                 accuracy_function,
                 directory,
                 f1score_function=None,
                 best_model=None,
                 predictor=None,
                 save_model_weights=True,
    ):

        self.name  = name
        self.config = config
        self.ROOT_DIR = directory

        self.log = logging.getLogger('{}.{}.{}'.format(__name__,
                                                       self.__class__.__name__,
                                                       self.name))

        self.model = model

        self.feed = feed

        self.predictor = predictor

        self.accuracy_function = accuracy_function if accuracy_function else self._default_accuracy_function
        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.f1score_function = f1score_function
        

        self.__build_stats()

        self.save_model_weights = save_model_weights
        self.best_model = (0.000001, self.model.cpu().state_dict())
        try:
            f = '{}/{}_best_model_accuracy.txt'.format(self.ROOT_DIR, self.name)
            if os.path.isfile(f):
                self.best_model = (float(open(f).read().strip()), self.model.cpu().state_dict())
                self.log.info('loaded last best accuracy: {}'.format(self.best_model[0]))
        except:
            log.exception('no last best model')

                        
        self.best_model_criteria = self.accuracy
        self.save_best_model()

        if self.config.CONFIG.cuda:
            self.model.cuda()
                        

    def do_every_checkpoint(self, epoch, early_stopping=True):

        self.model.eval()
        for j in tqdm(range(self.feed.num_batch), desc='Tester.{}'.format(self.name)):
            input_ = self.feed.next_batch()
            idxs, inputs, targets = input_
            sequence = inputs[0].transpose(0,1)
            _, batch_size = sequence.size()
            
            state = self.model.initial_hidden(batch_size)
            loss, accuracy = Var(self.config, [0]), Var(self.config, [0])
            output = sequence[0]
            outputs = []
            for ti in range(1, sequence.size(0) - 1):
                output = self.model(output, state)
                loss += self.loss_function(ti, output, input_)
                accuracy += self.accuracy_function(ti, output, input_)
                output, state = output
                output = output.max(1)[1]
                outputs.append(output)
                
            self.test_loss.cache(loss.item())
            if ti == 0: ti = 1
            self.accuracy.cache(accuracy.item()/ti)
            #print('====', self.test_loss, self.accuracy)

        self.log.info('= {} =loss:{}'.format(epoch, self.test_loss.epoch_cache))
        self.log.info('- {} -accuracy:{}'.format(epoch, self.accuracy.epoch_cache))

        if self.best_model[0] < self.accuracy.epoch_cache.avg:
            self.log.info('beat best model...')
            last_acc = self.best_model[0]
            self.best_model = (self.accuracy.epoch_cache.avg,
                               (self.model.state_dict())
                               
            )
            self.save_best_model()
            
            if self.config.CONFIG.cuda:
                self.model.cuda()

            if self.predictor and self.best_model[0] > 0.75:
                log.info('accuracy is greater than 0.75...')
                if ((
                        self.best_model[0] >= self.config.CONFIG.ACCURACY_THRESHOLD and
                        ( 5*(self.best_model[0] - last_acc) >
                          self.config.CONFIG.ACCURACY_IMPROVEMENT_THRESHOLD))
                    or (self.best_model[0] - last_acc) 
                    > self.config.CONFIG.ACCURACY_IMPROVEMENT_THRESHOLD
                ):
                    
                    self.predictor.run_prediction(self.accuracy.epoch_cache.avg)
                

        self.test_loss.clear_cache()
        self.accuracy.clear_cache()
        
        for m in self.metrics:
            m.write_to_file()
            
        if early_stopping:
            return self.loss_trend()
    
    
class Predictor(object):
    def __init__(self, name, model,
                 feed,
                 repr_function,
                 directory,
                 
                 *args, **kwargs):
        self.name = name
        self.ROOT_DIR = directory
                           
        self.model = model
        self.repr_function = repr_function

        self.log = logging.getLogger('{}.{}.{}'.format(__name__, self.__class__.__name__, self.name))

        self.feed = feed

    def predict(self,  batch_index=0, max_decoder_len=10):
        log.debug('batch_index: {}'.format(batch_index))
        idxs, i, *__ = self.feed.nth_batch(batch_index)
        outputs = []
        self.model.eval()
        input_ = self.feed.next_batch()
        idxs, inputs, targets = input_
        sequence = inputs[0].transpose(0,1)
        _, batch_size = sequence.size()
        
        state = self.model.initial_hidden(batch_size)
        loss = 0
        output = sequence[0]
        for ti in range(1, sequence.size(0) - 1):
            output = self.model(output, state)
            output, state = output
            output = output.max(1)[1]
            outputs.append(output)
                        
        results = ListTable()
        outputs = torch.stack(outputs)
        result = self.repr_function(outputs, input_)
        results.extend(result)
        return outputs, results

    def run_prediction(self, accuracy):        
        dump = open('{}/results/{}_{:0.4f}.csv'.format(self.ROOT_DIR, self.name, accuracy), 'w')
        self.log.info('on {}th eon'.format(accuracy))
        results = ListTable()
        for ri in tqdm(
                range(self.feed.num_batch),
                desc='running prediction at accuracy: {:0.4f}'.format(accuracy)):
            output, _results = self.predict(ri)
            results.extend(_results)
        dump.write(repr(results))
        dump.close()
