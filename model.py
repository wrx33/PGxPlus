# Define model
import torch.nn as nn
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
import warnings
warnings.filterwarnings("ignore")

class ClinicalModelDynamic(nn.Module):
    def __init__(self, modal_dims, drug_vocab_size, hidden_dim, n_heads, n_layers, num_classes=1, num_tasks=1):
        super(ClinicalModelDynamic, self).__init__()
        self.modal_names = list(modal_dims.keys())
        self.modal_extractors = nn.ModuleDict()

        # define modal extractors
        for modal, input_dim in modal_dims.items():
            if modal == 'gene':
                self.modal_extractors[modal] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim)
                )
            elif modal == 'demo':
                self.modal_extractors[modal] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim)
                )
            elif modal == 'exam':
                self.modal_extractors[modal] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim)
                )
            elif modal == 'scale':
                self.modal_extractors[modal] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim)
                )
            else:
                self.modal_extractors[modal] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim)
                )
        
        # Fusion layer
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * len(modal_dims), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # define cross-modal attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads)
        
        # define Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # # define output layer
        # self.output_layer = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.BatchNorm1d(hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim//2, num_classes),
        #     # nn.Softmax(dim=1)  # Multi-class classification
        # )
        
        self.multi_task_output_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.BatchNorm1d(hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim//2, num_classes),
            ) for _ in range(num_tasks)
        ])
        
        
    def forward(self, inputs):
        modal_features = []
        for modal, extractor in self.modal_extractors.items():
            if modal == 'age' or modal == 'duration' or modal == 'bmi':# or modal == 'description1' or modal == 'description2' or modal == 'description3':
                modal_input = inputs[modal].unsqueeze(-1)
                features = extractor(modal_input)
            else:
                features = extractor(inputs[modal].squeeze(1))
            modal_features.append(features)
        
        # Concatenate and fuse modal features
        combined_features = torch.cat(modal_features, dim=-1)
        combined_features = self.fusion_mlp(combined_features)
        
        # Cross-modal attention
        encoded_features = self.transformer_encoder(combined_features)
        
        # Output layer
        # output = self.output_layer(encoded_features)
        multi_outputs = [output_layer(encoded_features) for output_layer in self.multi_task_output_layer]
        return multi_outputs
    

# class ClinicalModelDynamic(nn.Module):
#     def __init__(self, modal_dims, drug_vocab_size, hidden_dim, n_heads, n_layers, num_classes=1):
#         super(ClinicalModelDynamic, self).__init__()
#         self.modal_names = list(modal_dims.keys())
#         self.modal_extractors = nn.ModuleDict()
#         # self.modal_weights = nn.Parameter(torch.ones(len(self.modal_names)), requires_grad=True)

#         # define modal extractors
#         for modal, input_dim in modal_dims.items():
#             if modal == 'gene':
#                 self.modal_extractors[modal] = nn.Sequential(
#                     # nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
#                     nn.Linear(input_dim, hidden_dim),
#                     nn.ReLU(),
#                     # nn.Flatten(),
#                     nn.Linear(hidden_dim, hidden_dim)
#                 )
#             elif modal == 'drug':
#                 self.modal_extractors[modal] = nn.Embedding(drug_vocab_size, hidden_dim)
#             else:
#                 self.modal_extractors[modal] = nn.Sequential(
#                     nn.Linear(input_dim, hidden_dim),
#                     nn.ReLU(),
#                     nn.Linear(hidden_dim, hidden_dim)
#                 )
        
#         # 融合层
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(hidden_dim * len(modal_dims), hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
        
#         # define cross-modal attention
#         self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads)
        
#         # define Trasnformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
#         # define output layer
#         self.output_layer = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim//2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim//2, num_classes),
#             # nn.Sigmoid() # 二分类
#             nn.Softmax() # 多分类
#         )
        
#     def forward(self, inputs):
#         modal_features = []
#         for modal, extractor in self.modal_extractors.items():
#             if modal == 'gene':
#                 modal_input = inputs[modal]
#                 features = extractor(modal_input)
#             elif modal == 'drug':
#                 features = extractor(inputs[modal]).mean(dim=1)
#             else:
#                 features = extractor(inputs[modal])
#             modal_features.append(features)
        
#         # 模态特征拼接并融合
#         combined_features = torch.cat(modal_features, dim=-1)
#         combined_features = self.fusion_mlp(combined_features)
        
#         # Cross-Attention
#         # query = combined_features.unsqueeze(0)  # 融合特征作为 Query
#         # key_value = torch.stack(modal_features, dim=0)  # 各模态特征作为 Key 和 Value
#         # cross_features, _ = self.cross_attention(query=query, key=key_value, value=key_value)

        
#         # cross-modal attention
#         # cross_features, _ = self.cross_attention(
#         #     query=combined_features.unsqueeze(0),
#         #     key=combined_features.unsqueeze(0),
#         #     value=combined_features.unsqueeze(0)
#         # )
        
#         # self-attention
#         # encoded_features = self.transformer_encoder(cross_features.mean(dim=0))
#         encoded_features = self.transformer_encoder(combined_features)
        
#         # output layer
#         # output = self.output_layer(cross_features.mean(dim=0))
#         output = self.output_layer(combined_features)
#         return output