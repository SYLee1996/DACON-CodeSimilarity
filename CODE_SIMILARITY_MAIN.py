import os 
import copy 
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import logging, AutoTokenizer, RobertaTokenizer

logging.set_verbosity_error()

import warnings
warnings.filterwarnings(action='ignore')

from CODE_SIMILARITY_UTILS import (
    CustomDataset, UniformLengthBatchingSampler, Custom_collate_fn,
    CosineAnnealingWarmUpRestarts, 
    SmoothCrossEntropyLoss, 
    EarlyStopping, 
    score_function)
from CODE_SIMILARITY_MODEL import Network


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Training', add_help=False)

    # Model parameters
    parser.add_argument('--model_name', default='cross-encoder/ms-marco-MiniLM-L-12-v2', type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--drop_out', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=10, type=int)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--label_smoothing', default=0.3, type=float)


    # Optimizer parameters
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_t', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', type=str)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--weight_decay', default=0.001, type=float)


    # Training parameters
    parser.add_argument('--train_data', default='data_path', type=str)
    parser.add_argument('--valid_data', default='data_path', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--text', default='default', type=str)
    parser.add_argument('--device', default='0,1,2,3', type=str)

    return parser


def main(args):

    seed = 10
    suffix = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M")

    config = {
        # Model parameters
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'max_seq_len': args.max_seq_len,
        'drop_out': args.drop_out,
        'max_grad_norm': args.max_grad_norm,
        'num_labels': args.num_labels,
        'label_smoothing': args.label_smoothing,
        
        # Optimizer parameters
        'optimizer': args.optimizer,
        'lr': args.lr,
        'lr_t': args.lr_t,
        'lr_scheduler': args.lr_scheduler,
        'gamma': args.gamma,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        
        # Training parameters
        'train_data': args.train_data,
        'valid_data': args.valid_data,
        'mode': args.mode,
        'epochs': args.epochs,
        'num_workers': args.num_workers,
        'text': args.text,
        'device': args.device,
        }
    
    model_save_name='./RESULTS/'+config['text']+"_"+suffix+"("+ str(config['batch_size'])+"_"+\
                                                                str(config['max_seq_len'])+"_"+\
                                                                str(config['drop_out'])+"_"+\
                                                                str(config['max_grad_norm'])+"_"+\
                                                                str(config['num_labels'])+"__"+\
                                                                str(config['label_smoothing'])+"__"+\
                                                                str(config['optimizer'])+"_"+\
                                                                str(config['lr'])+"_"+\
                                                                str(config['lr_t'])+"_"+\
                                                                str(config['lr_scheduler'])+"_"+\
                                                                str(config['gamma'])+"_"+\
                                                                str(config['patience'])+"_"+\
                                                                str(config['weight_decay'])+")"
                                                            
    config['model_save_name'] = model_save_name
    print('model_save_name: '+config['model_save_name'].split("/")[-1])
    # -------------------------------------------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    config['device'] = device

    # -------------------------------------------------------------------------------------------
    
    # Dataload
    train_AUG = pd.read_csv(config['train_data'])
    valid_AUG = pd.read_csv(config['valid_data'])
    
    if (config['model_name'] == 'cross-encoder/ms-marco-MiniLM-L-12-v2') | \
        (config['model_name'] == "cross-encoder/ms-marco-electra-base") | \
        (config['model_name'] == "huggingface/CodeBERTa-small-v1") | \
        (config['model_name'] == "sentence-transformers/paraphrase-xlm-r-multilingual-v1"):
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
    else :
        tokenizer = RobertaTokenizer.from_pretrained(config['model_name'], use_fast=True)
    
    train_AUG.dropna(inplace=True)
    valid_AUG.dropna(inplace=True)
        
    train_AUG.reset_index(drop=True, inplace=True)
    valid_AUG.reset_index(drop=True, inplace=True)

    # Train
    train_set = CustomDataset(data=train_AUG)
    sampler = UniformLengthBatchingSampler(data_source=train_set, config=config)
    collate_fn = Custom_collate_fn(tokenizer=tokenizer, config=config)
    
    Train_loader=DataLoader(dataset=train_set,
                            sampler=sampler,
                            collate_fn=collate_fn,
                            pin_memory=True, num_workers=config['num_workers'], 
                            prefetch_factor=config['batch_size']*2,
                            )
    
    # Valid
    valid_set = CustomDataset(data=valid_AUG)
    sampler = UniformLengthBatchingSampler(data_source=valid_set, config=config)
    collate_fn = Custom_collate_fn(tokenizer=tokenizer, config=config)
    Valid_loader=DataLoader(dataset=train_set,
                            sampler=sampler,
                            collate_fn=collate_fn,
                            pin_memory=True, num_workers=config['num_workers'], 
                            prefetch_factor=config['batch_size']*2,
                            )

    model = Network(config).to(config['device'])
    model = nn.DataParallel(model).to(config['device'])

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = config['drop_out']
        
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
        'weight_decay': config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0}
        ]
    
    if config['lr_scheduler'] == 'CosineAnnealingLR':
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_t'], eta_min=0)
        
    elif config['lr_scheduler'] == 'CosineAnnealingWarmUpRestarts':
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=0)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=config['lr_t'], eta_max=config['lr'], gamma=config['gamma'], T_mult=1, T_up=0)
    
    criterion = SmoothCrossEntropyLoss(smoothing=config['label_smoothing']).to(config['device'])
    scaler = torch.cuda.amp.GradScaler() 
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    early_stopping_loss = EarlyStopping(patience=config['patience'], mode='min')
    
    best=0.5
    best_loss=10
    each_fold_valid_loss, each_fold_valid_f1 = [], []
    epochs = config['epochs']
    
    for epoch in range(epochs):
        valid_loss, valid_pred, valid_real = 0, [], []
        
        model.train()
        tqdm_dataset = tqdm(enumerate(Train_loader), total=len(Train_loader))

        iii = 0
        for batch_id, batch in tqdm_dataset:
            if iii == 100:
                break
            optimizer.zero_grad()
            try:
                ids = torch.tensor(batch['input_ids'], dtype=torch.long, device=config['device'])
                atts = torch.tensor(batch['attention_mask'], dtype=torch.long, device=config['device'])
                labels = torch.tensor(batch['similar'], dtype=torch.long, device=config['device'])
                token_type = torch.tensor(batch['token_type_ids'], dtype=torch.long, device=config['device'])
                
                train_inputs = {'input_ids': ids,
                                'token_type_ids': token_type,
                                'attention_mask': atts}
            except:
                train_inputs = {'input_ids': ids,
                                'attention_mask': atts}
            
            with torch.cuda.amp.autocast():    
                pred = model(**train_inputs)
                
            loss = criterion(pred, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            
            train_pred = torch.max(pred, 1)[1].detach().cpu().numpy().tolist()
            train_real = labels.detach().cpu().numpy().tolist()
            tqdm_dataset.set_postfix({
                'F-1' : '{:06f}'.format(score_function(train_real, train_pred))
                })
            # iii+=1
        scheduler.step()

        model.eval()
        tqdm_valid_dataset =  tqdm(enumerate(Valid_loader), total=len(Valid_loader))
        iiii=0
        for batch_id, val_batch in tqdm_valid_dataset:
            if iiii==100:
                break
            with torch.no_grad():
                try:
                    val_ids = torch.tensor(val_batch['input_ids'], dtype=torch.long, device=config['device'])
                    val_atts = torch.tensor(val_batch['attention_mask'], dtype=torch.long, device=config['device'])
                    val_labels = torch.tensor(val_batch['similar'], dtype=torch.long, device=config['device'])
                    val_token_type = torch.tensor(val_batch['token_type_ids'], dtype=torch.long, device=config['device'])
                    
                    valid_inputs = {'input_ids': val_ids,
                                    'token_type_ids': val_token_type,
                                    'attention_mask': val_atts}
                except:
                    valid_inputs = {'input_ids': val_ids,
                                    'attention_mask': val_atts}

                val_pred = model(**valid_inputs)                   
                val_loss = criterion(val_pred, val_labels)

            valid_loss += val_loss.item()
            valid_pred += torch.max(val_pred, 1)[1].detach().cpu().numpy().tolist()
            valid_real += val_labels.detach().cpu().numpy().tolist()
            tqdm_valid_dataset.set_postfix({
                'F-1' : '{:06f}'.format(score_function(valid_real, valid_pred))
                })
            # iiii+=1
            
        valid_loss = valid_loss/len(Valid_loader)
        valid_f1 = score_function(valid_real, valid_pred)
        each_fold_valid_loss.append(valid_loss)
        each_fold_valid_f1.append(valid_f1)
        
        print_best = 0    
        if (each_fold_valid_f1[-1] >= best) | (each_fold_valid_loss[-1] <= best_loss):
            difference = each_fold_valid_f1[-1] - best
            if (each_fold_valid_f1[-1] >= best):
                best = each_fold_valid_f1[-1] 
            if (each_fold_valid_loss[-1] <= best_loss):
                best_loss = each_fold_valid_loss[-1]
            
            pprint_best = each_fold_valid_f1[-1]
            pprint_best_loss = each_fold_valid_loss[-1]
            
            best_idx = epoch+1
            model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
            best_model_wts = copy.deepcopy(model_state_dict)
            
            # load and save best model weights
            model.module.load_state_dict(best_model_wts)
            torch.save(best_model_wts, config['model_save_name'] + ".pt")
            print_best = '==> best model saved %d epoch / acc: %.5f  loss: %.5f  /  difference %.5f'%(best_idx, pprint_best, pprint_best_loss, difference)

        print(f'epoch : {epoch+1}/{epochs}')
        print(f'VALID_Loss : {valid_loss:.5f}    VALID_F1 : {valid_f1:.5f}    BEST : {pprint_best:.5f}    BEST_LOSS : {pprint_best_loss:.5f}')
        print('\n') if type(print_best)==int else print(print_best,'\n')

        del loss; del val_loss; del train_pred; del train_real; del valid_pred; del valid_real
        torch.cuda.empty_cache()

        if early_stopping.step(torch.tensor(each_fold_valid_f1[-1])) & early_stopping_loss.step(torch.tensor(each_fold_valid_loss[-1])):
            break
        
    print("VALID Loss: ", pprint_best_loss, ", VALID F1: ", pprint_best)
    print(config['model_save_name'].split("/")[-1] + ' is saved!')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)


