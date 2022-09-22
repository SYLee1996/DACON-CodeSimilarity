import os 
import random
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

from CODE_SIMILARITY_UTILS import TestDataset
from CODE_SIMILARITY_MODEL import Network


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Inference', add_help=False)

    # Model parameters
    parser.add_argument('--model_save_name_list', nargs='+', default='load_models', type=str)
    parser.add_argument('--model_name_list', nargs='+', default='model_names', type=str)
    parser.add_argument('--model_name_16layer', nargs='+', default='model_boolean', type=str)
    parser.add_argument('--test_data', default='data_path', type=str)
    parser.add_argument('--save_name', default='save_name', type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--drop_out', default=0.0, type=float)
    parser.add_argument('--device', default='0,1,2,3', type=str)

    return parser


def main(args):

    seed = 10

    config = {
        # Model parameters
        'model_save_name_list': args.model_save_name_list,
        'model_name_list': args.model_name_list,
        'model_name_16layer': args.model_name_16layer,
        'test_data': args.test_data,
        'save_name': args.save_name,

        'batch_size': args.batch_size,
        'num_labels': args.num_labels,
        'max_seq_len': args.max_seq_len,
        'num_workers': args.num_workers,
        'drop_out': args.drop_out,
        'device': args.device
        }
    
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
    dataset = pd.read_csv(config['test_data'])
    
    bin_df = pd.DataFrame()

    for i, (model_save_name, model_name, model_layer) in enumerate(zip(config['model_save_name_list'], config['model_name_list'], config['model_name_16layer'])):
        config['model_save_name'] = model_save_name
        config['model_name'] = model_name
        config['model_layer'] = model_layer
        print(model_name)

        if (config['model_name'] == 'cross-encoder/ms-marco-MiniLM-L-12-v2') | \
            (config['model_name'] == "cross-encoder/ms-marco-electra-base") | \
            (config['model_name'] == "huggingface/CodeBERTa-small-v1") | \
            (config['model_name'] == "sentence-transformers/paraphrase-xlm-r-multilingual-v1"):
            tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
        else :
            tokenizer = RobertaTokenizer.from_pretrained(config['model_name'], use_fast=True)
        
        model = Network(config).to(device)
        model = nn.DataParallel(model).to(device)
        model_dict = torch.load('./RESULTS/'+config['model_save_name'] + ".pt")
        model.module.load_state_dict(model_dict) if torch.cuda.device_count() > 1 else model.load_state_dict(model_dict)
    
        # Test 
        test_set = TestDataset(data=dataset, 
                                tokenizer=tokenizer,
                                config=config)
        Test_loader =DataLoader(test_set, batch_size=config['batch_size'], pin_memory=True,
                                num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2, 
                                shuffle=False)
                                
        results = []
        for batch_id, batch in tqdm(enumerate(Test_loader), total=len(Test_loader)):
            try:
                ids = torch.tensor(batch['input_ids'], dtype=torch.long, device=config['device'])
                atts = torch.tensor(batch['attention_mask'], dtype=torch.long, device=config['device'])
                token_type = torch.tensor(batch['token_type_ids'], dtype=torch.long, device=config['device'])
                
                test_inputs = {'input_ids': ids,
                                'token_type_ids': token_type,
                                'attention_mask': atts}
            except:
                test_inputs = {'input_ids': ids,
                                'attention_mask': atts}
        
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = model(**test_inputs)

            results.extend(output.detach().cpu().numpy().tolist())    

        df = pd.DataFrame(results)  

        if i == 0:
            bin_df = df
        else: 
            bin_df = bin_df + df 

        del model; del tokenizer; del test_set; del Test_loader
        torch.cuda.empty_cache()

    result_df = bin_df.idxmax(axis="columns") 
    submission = pd.read_csv("/home/CODE_SIMILARITY/sample_submission.csv")
    submission['similar'] = result_df
    
    submission.to_csv("./RESULTS/{}.csv".format(config['save_name']), index=False)
    print(config['save_name']+ ".csv is saved!")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('test script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)


