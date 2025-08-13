import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class BertEncoder(nn.Module):
    def __init__(self, bert_model, output_dim=128):
        super(BertEncoder, self).__init__()
        self.bert = bert_model
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.init_weights()
    
    def init_weights(self):
        # Xavier initialization
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.linear(pooled_output)


class ClinicalDataset(Dataset):
    def __init__(self, data, all_data):
        self.data = data
        self.encoder_gene = OneHotEncoder(sparse=False,)
        self.encoder_demo = OneHotEncoder(sparse=False)
        self.scaler_scale = StandardScaler()
        self.scaler_exam = StandardScaler()
        # self.scaler_age = StandardScaler()
        # self.scaler_duration = StandardScaler()
        
        self.tokenizer = BertTokenizer.from_pretrained('./base_models/bert-base-uncased')
        self.bertmodel = BertModel.from_pretrained('./base_models/bert-base-uncased')
        self.bertencoder = BertEncoder(self.bertmodel, output_dim=128)

        # gene_data = [sample['gene'] for sample in data]
        # age_data = [[sample['age']] for sample in data]
        # demo_data = [sample['demo'] for sample in data]
        # duration_data = [[sample['duration']] for sample in data]
        # exam_data = [sample['exam'] for sample in data]
        # scale_data = [sample['scale'] for sample in data]
        
        self.encoder_gene.fit([sample['gene'] for sample in all_data])    # 基因特征的one-hot编码维度是219
        self.encoder_demo.fit([sample['demo'] for sample in all_data])    # demo特征的one-hot编码维度是16
        self.scaler_scale.fit([sample['scale'] for sample in all_data])   
        self.scaler_exam.fit([sample['exam'] for sample in all_data])
        # self.scaler_age.fit(age_data)
        # self.scaler_duration.fit(duration_data)
        
        self.min_age = np.min([[sample['age']] for sample in all_data])
        self.max_age = np.max([[sample['age']] for sample in all_data])
        self.min_duration = np.min([[sample['duration']] for sample in all_data])
        self.max_duration = np.max([[sample['duration']] for sample in all_data])

    def get_feature_dimensions_gene(self):
        return sum([category.shape[0] for category in self.encoder_gene.categories_])
    
    def get_feature_dimensions_demo(self):
        return sum([category.shape[0] for category in self.encoder_demo.categories_])
    
    def get_task_num(self):
        return len(self.data[0]['label'])
    
    def normalize_age(self, age):
        return(age - self.min_age) / (self.max_age - self.min_age)
    
    def normalize_duration(self, duration):
        return (duration - self.min_duration) / (self.max_duration - self.min_duration)
    
    def __len__(self):
        return len(self.data)

    def encode_description(self, description):
        inputs = self.tokenizer(description, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            encoded_description = self.bertencoder(inputs['input_ids'], inputs['attention_mask'])
        return encoded_description.squeeze()

    def __getitem__(self, idx):
        
        sample = self.data[idx]
        for key, val in sample.items():
            if key == 'gene':
                gene_input = self.encoder_gene.transform([val])
            elif key == 'demo':
                demo_input = self.encoder_demo.transform([val])
            elif key == 'age':
                age_input = self.normalize_age(val)
            elif key == 'duration':
                duration_input = self.normalize_duration(val)
            elif key == 'exam':
                exam_input = self.scaler_exam.transform([val])
            elif key == 'scale':
                scale_input = self.scaler_scale.transform([val])
            elif key == 'description1':
                if val == 0:
                    description1_input = self.encode_description('无明显诱因')
                else:
                    description1_input = self.encode_description(val)
            elif key == 'description2':
                if val == 0:
                    description2_input = self.encode_description('不详')
                else:
                    description2_input = self.encode_description(val)  
            elif key == 'description3':
                if val == 0:
                    description3_input = self.encode_description('不详')
                else:
                    description3_input = self.encode_description(val)
            elif key == 'medication_history':
                    binary_list = [int(char) for char in val]
                    medication_input = np.array(binary_list).reshape(1, -1)

        inputs = {
            'gene': torch.tensor(gene_input, dtype=torch.float32),
            'age': torch.tensor(age_input, dtype=torch.float32),
            'demo': torch.tensor(demo_input, dtype=torch.float32),
            'duration': torch.tensor(duration_input, dtype=torch.float32),
            'exam': torch.tensor(exam_input, dtype=torch.float32),
            'scale': torch.tensor(scale_input, dtype=torch.float32),
            'description1': description1_input,
            'description2': description2_input,
            'description3': description3_input,
            'medication_history': torch.tensor(medication_input, dtype=torch.float32)
        }
        # inputs = {key: torch.tensor(val, dtype=torch.float32) for key, val in sample.items() if key != 'label'}
        label = torch.tensor(sample['label'], dtype=torch.float32)
        return inputs, label


class ClinicalDataset_simple(Dataset):
    def __init__(self, data):
        self.data = data
    
    def get_feature_dimensions_gene(self):
        return self.data[0][0]['gene'].shape[1]
    
    def get_feature_dimensions_demo(self):
        return self.data[0][0]['demo'].shape[1]
    
    def get_feature_dimensions_mh(self):
        return self.data[0][0]['medication_history'].shape[1]

    def get_feature_dimensions_od(self):
        return self.data[0][0]['other_disease'].shape[1]
    
    def get_task_num(self):
        return self.data[0][1].shape[0]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        sample = self.data[idx]
        
        return sample[0], sample[1]