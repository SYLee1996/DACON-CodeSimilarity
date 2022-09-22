import copy
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, RobertaModel


class Network(torch.nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.model_name = config['model_name']
        self.num_class = config['num_labels']
        self.drop_rate = config['drop_out']
        self.layer = config['model_layer']
        
        self.dropout = nn.Dropout(self.drop_rate)
        
        if self.model_name == "microsoft/codebert-base":
            model_config = AutoConfig.from_pretrained(self.model_name)
            if self.layer == 'True':
                print('True')
                model_config.update({"num_attention_heads": 12,
                                    "num_hidden_layers": 16})  
            self.model = RobertaModel.from_pretrained(self.model_name, config=model_config)
            self.out_layer = nn.Linear(768, self.num_class)
        
        
        elif self.model_name == "microsoft/graphcodebert-base":
            model_config = AutoConfig.from_pretrained(self.model_name)
            if self.layer == 'True':
                print('True')
                model_config.update({"num_attention_heads": 12,
                                    "num_hidden_layers": 16})  
            self.model = RobertaModel.from_pretrained(self.model_name, config=model_config)
            self.out_layer = nn.Linear(768, self.num_class)
        
        
        elif self.model_name == "cross-encoder/ms-marco-electra-base":
            model_config = AutoConfig.from_pretrained(self.model_name)
            if self.layer == 'True':
                print('True')
                model_config.update({"num_attention_heads": 12,
                                    "num_hidden_layers": 15})  
            self.model = AutoModel.from_pretrained(self.model_name, config=model_config)           
            self.out_layer = nn.Linear(768, self.num_class)
            
            
        elif self.model_name == "huggingface/CodeBERTa-small-v1":
            model_config = AutoConfig.from_pretrained(self.model_name)
            if self.layer == 'True':
                print('True')
                model_config.update({"num_attention_heads": 12,
                                    "num_hidden_layers": 16})  
            self.model = AutoModel.from_pretrained(self.model_name, config=model_config)           
            self.out_layer = nn.Linear(768, self.num_class)
            
            
        elif self.model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2":
            model_config = AutoConfig.from_pretrained(self.model_name)
            if self.layer == 'True':
                model_config.update({"num_attention_heads": 12,
                                    "num_hidden_layers": 16})  
            self.model = AutoModel.from_pretrained(self.model_name, config=model_config)   
            self.out_layer = nn.Linear(384, self.num_class)
                    

    def forward(self, **x):
        attention_mask = x['attention_mask']

        x = self.model(**x)[0]
        x_pool = mean_pooling(x, attention_mask)
        x_pool = torch.flatten(x_pool, start_dim=1)
        output = self.dropout(self.out_layer(x_pool))
        return output

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
