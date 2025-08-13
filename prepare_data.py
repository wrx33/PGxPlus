import pandas as pd
import numpy as np
import json
import re
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
import argparse
warnings.filterwarnings("ignore")

class BertEncoder(nn.Module):
    def __init__(self, bert_model, output_dim=128):
        super(BertEncoder, self).__init__()
        self.bert = bert_model
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.linear(pooled_output)

class ClinicalDataset(Dataset):
    def __init__(self, data, output_path):
        self.data = data
        self.output_path = output_path
        self.encoder_gene = OneHotEncoder(sparse_output=False)
        self.encoder_mh = OneHotEncoder(sparse_output=False)
        self.encoder_demo = OneHotEncoder(sparse_output=False)
        self.encoder_od = OneHotEncoder(sparse_output=False)
        
        self.scaler_hamd17 = StandardScaler()
        self.scaler_scale = StandardScaler()
        self.scaler_exam = StandardScaler()
        
        self.tokenizer = BertTokenizer.from_pretrained('./base_models/bert-base-uncased')
        self.bertmodel = BertModel.from_pretrained('./base_models/bert-base-uncased')
        self.bertencoder = BertEncoder(self.bertmodel, output_dim=128)
        
        self.encoder_gene.fit([sample['gene'] for sample in data])  
        self.encoder_demo.fit([sample['demo'] for sample in data])  
        self.encoder_od.fit([sample['other_disease'] for sample in data])
        self.encoder_mh.fit([sample['medication_history'] for sample in data])
        
        self.scaler_hamd17.fit([[sample['hamd17']] for sample in data])
        self.scaler_scale.fit([sample['scale'] for sample in data])   
        self.scaler_exam.fit([sample['exam'] for sample in data])

        self.min_age = np.min([[sample['age']] for sample in data])
        self.max_age = np.max([[sample['age']] for sample in data])
        
        self.min_bmi = np.min([[sample['bmi']] for sample in data])
        self.max_bmi = np.max([[sample['bmi']] for sample in data])
        
        self.min_duration = np.min([[sample['duration']] for sample in data])
        self.max_duration = np.max([[sample['duration']] for sample in data])
        
    def get_feature_dimensions_gene(self):
        return sum([category.shape[0] for category in self.encoder_gene.categories_])
    
    def get_feature_dimensions_mh(self):
        return sum([category.shape[0] for category in self.encoder_mh.categories_])
    
    def get_feature_dimensions_demo(self):
        return sum([category.shape[0] for category in self.encoder_demo.categories_])

    def get_feature_dimensions_od(self):
        return sum([category.shape[0] for category in self.encoder_od.categories_])
    
    def normalize_age(self, age):
        return(age - self.min_age) / (self.max_age - self.min_age)
    
    def normalize_bmi(self, bmi):
        return(bmi - self.min_bmi) / (self.max_bmi - self.min_bmi)
    
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
            elif key == 'bmi':
                age_input = self.normalize_bmi(val)
            elif key == 'duration':
                duration_input = self.normalize_duration(val)
            elif key == 'exam':
                exam_input = self.scaler_exam.transform([val])
            elif key == 'scale':
                scale_input = self.scaler_scale.transform([val])
            elif key == 'other_disease':
                od_input = self.encoder_od.transform([val])
            elif key == 'hamd17':
                hamd17_input = self.scaler_hamd17.transform([[val]])
            elif key == 'medication_history':
                medication_input = self.encoder_mh.transform([val])

        inputs = {
            'gene': torch.tensor(gene_input, dtype=torch.float32),
            'age': torch.tensor(age_input, dtype=torch.float32),
            'bmi': torch.tensor(age_input, dtype=torch.float32), 
            'demo': torch.tensor(demo_input, dtype=torch.float32),
            'duration': torch.tensor(duration_input, dtype=torch.float32),
            'exam': torch.tensor(exam_input, dtype=torch.float32),
            'scale': torch.tensor(scale_input, dtype=torch.float32),
            'other_disease': torch.tensor(od_input, dtype=torch.float32),
            'hamd17': torch.tensor(hamd17_input, dtype=torch.float32),
            'medication_history': torch.tensor(medication_input, dtype=torch.float32)
        }
        label = torch.tensor(sample['label'], dtype=torch.float32)
        return inputs, label

    def save_data(self):
        save_data = []
        for i in range(len(self.data)):
            print(i)
            sample = self.data[i]
            for key, val in sample.items():
                if key == 'gene':
                    gene_input = self.encoder_gene.transform([val])
                elif key == 'demo':
                    demo_input = self.encoder_demo.transform([val])
                elif key == 'age':
                    age_input = self.normalize_age(val)
                elif key == 'bmi':
                    bmi_input = self.normalize_bmi(val)
                elif key == 'duration':
                    duration_input = self.normalize_duration(val)
                elif key == 'exam':
                    exam_input = self.scaler_exam.transform([val])
                elif key == 'scale':
                    scale_input = self.scaler_scale.transform([val])
                elif key == 'other_disease':
                    od_input = self.encoder_od.transform([val])
                elif key == 'hamd17':
                    hamd17_input = self.scaler_hamd17.transform([[val]])
                elif key == 'medication_history':
                    medication_input = self.encoder_mh.transform([val])
                
            inputs = {
                'gene': torch.tensor(gene_input, dtype=torch.float32),
                'age': torch.tensor(age_input, dtype=torch.float32),
                'bmi': torch.tensor(bmi_input, dtype=torch.float32), 
                'demo': torch.tensor(demo_input, dtype=torch.float32),
                'duration': torch.tensor(duration_input, dtype=torch.float32),
                'exam': torch.tensor(exam_input, dtype=torch.float32),
                'scale': torch.tensor(scale_input, dtype=torch.float32),
                'other_disease': torch.tensor(od_input, dtype=torch.float32),
                'hamd17': torch.tensor(hamd17_input, dtype=torch.float32),
                'medication_history': torch.tensor(medication_input, dtype=torch.float32)
            }

            label_raw = torch.tensor(sample['label'], dtype=torch.float32)
            label = label_raw.clone()
            label[label_raw == 1] = 3
            label[label_raw == 3] = 1
            save_data.append((inputs, label))
        torch.save(save_data, self.output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input and output paths.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output file.")
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    raw_data = pd.read_excel(input_path)
    raw_data = raw_data.fillna(-1)

    data = []
    for index, row in raw_data.iterrows():
        sample = {
            'gene': row[1:71].tolist(),
            'medication_history': row[71:91].tolist(),
            'age': row[91],
            'bmi': round(row[92],2),
            'duration': row[93],
            'demo': row[94:101].tolist(),
            'exam': row[101:110].tolist(),
            'other_disease': row[110:119].tolist(),
            'hamd17': row[119],
            'scale': row[120:137].tolist(),
            'label': row[137:]
        }
        data.append(sample)

    dataset = ClinicalDataset(data, output_path)
    dataset.save_data()
