import math
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import _LRScheduler


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        if (self.config['model_name'] == 'cross-encoder/ms-marco-MiniLM-L-12-v2') | \
            (self.config['model_name'] == 'cross-encoder/ms-marco-electra-base'):
            encoding_result =self.tokenizer.encode_plus(record['code1'], record['code2'], 
                                            max_length=self.config['max_seq_len'], 
                                            padding='max_length',
                                            truncation=True
                                            )
            return {'input_ids': np.array(encoding_result['input_ids'], dtype=int),
                    'attention_mask': np.array(encoding_result['attention_mask'], dtype=int),
                    'token_type_ids': np.array(encoding_result['token_type_ids'], dtype=int)
                    }
        else:
            record['SEP'] = self.tokenizer.sep_token
            list_data = record['code1'] + record['SEP'] + record['code2']

            encoding_result =self.tokenizer.encode_plus(list_data, 
                                                        max_length=self.config['max_seq_len'], 
                                                        padding='max_length',
                                                        truncation=True
                                                        )
            return {'input_ids': np.array(encoding_result['input_ids'], dtype=int),
                    'attention_mask': np.array(encoding_result['attention_mask'], dtype=int),
                    }


class CustomDataset(Dataset):
    def __init__(self, data):
        super(CustomDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        return record


class UniformLengthBatchingSampler(Sampler):
    def __init__(self, data_source: Dataset, config: None):
        super(UniformLengthBatchingSampler, self).__init__(data_source)
        self.data_source = data_source      
        self.config = config      
        
        ids = sorted(range(len(self.data_source)), key=lambda x: len(self.data_source.data['code1'][x]+self.data_source.data['code2'][x]))
        self.bins = [ids[i:i + self.config['batch_size']] for i in range(0, len(ids), self.config['batch_size'])]
        
        if self.config['mode'] == 'train':
            np.random.shuffle(self.bins)
                
    def __iter__(self):
        for ids in self.bins:
            yield ids

    def __len__(self):
        return len(self.bins)


class Custom_collate_fn(object):
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
    
    def __call__(self, batch):
        batch = batch[0]
        
        if (self.config['model_name'] == 'cross-encoder/ms-marco-MiniLM-L-12-v2') | \
            (self.config['model_name'] == 'cross-encoder/ms-marco-electra-base'):
            encoding_result = self.tokenizer(list(batch['code1']),list(batch['code2']), padding=False, truncation=True)
            max_len = max([len(x) for x in encoding_result['input_ids']])
            
            batch_padded_inputs = []
            batch_token_type_ids = []
            batch_attn_masks = []
            
            for input_ids, token_type_ids in zip(encoding_result.input_ids, encoding_result.token_type_ids):
                
                # pad input_ids
                num_pads = max_len - len(input_ids)
                padded_input = input_ids + [self.tokenizer.pad_token_id] * num_pads
                
                # pad token_type_ids
                padded_token_type_ids = token_type_ids + [1] * num_pads
                
                # # Define the attention mask
                attn_mask = [1] * len(input_ids) + [0] * num_pads
                
                # # Add the padded results to the batch
                batch_padded_inputs.append(padded_input)
                batch_token_type_ids.append(padded_token_type_ids)
                batch_attn_masks.append(attn_mask)
            
            if self.config['mode'] == 'train':                         
                return {'input_ids': np.array(batch_padded_inputs, dtype=int),
                        'token_type_ids': np.array(batch_token_type_ids, dtype=int),
                        'attention_mask': np.array(batch_attn_masks, dtype=int),
                        'similar': np.array(batch['similar'], dtype=int)}
            else:
                return {'input_ids': np.array(batch_padded_inputs, dtype=int),
                        'token_type_ids': np.array(batch_token_type_ids, dtype=int),
                        'attention_mask': np.array(batch_attn_masks, dtype=int),
                        }
            
        else:
            batch['SEP'] = self.tokenizer.sep_token
            batch_data = batch['code1'] + batch['SEP'] + batch['code2']
            encoding_result = self.tokenizer(list(batch_data), padding=False, truncation=True)
                    
            max_len = max([len(x) for x in encoding_result['input_ids']])
            
            batch_padded_inputs = []
            batch_attn_masks = []
                
            for input_ids in encoding_result.input_ids:
                
                # pad tokens
                num_pads = max_len - len(input_ids)
                padded_input = input_ids + [self.tokenizer.pad_token_id] * num_pads
                
                # # Define the attention mask
                attn_mask = [1] * len(input_ids) + [0] * num_pads
                
                # # Add the padded results to the batch
                batch_padded_inputs.append(padded_input)
                batch_attn_masks.append(attn_mask)
                
            if self.config['mode'] == 'train':                         
                return {'input_ids': np.array(batch_padded_inputs, dtype=int),
                        'attention_mask': np.array(batch_attn_masks, dtype=int),
                        'similar': np.array(batch['similar'], dtype=int)}
            else:
                return {'input_ids': np.array(batch_padded_inputs, dtype=int),
                        'attention_mask': np.array(batch_attn_masks, dtype=int),
                        }


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.1):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score
