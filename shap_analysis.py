# explain the importance of features using SHAP
import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from data import *
from model import ClinicalModelDynamic
from loss import *
from shap_explain import *
from config import sweep_config
import joblib
import os
import argparse
import wandb
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_data(data_path, random_seed):
    data = torch.load(data_path)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)
    train_dataset = ClinicalDataset_simple(train_data)
    test_dataset = ClinicalDataset_simple(test_data)
    train_loader = DataLoader(train_dataset, batch_size=2500, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2500, shuffle=False)
    return train_dataset, train_loader, test_dataset, test_loader

def load_model(model_path, dataset, device):
    # Hyperparameters
    modal_dims = {
        'gene': dataset.get_feature_dimensions_gene(),
        'medication_history':  dataset.get_feature_dimensions_mh(),
        'age': 1,
        'bmi': 1,
        'duration': 1,
        'demo': dataset.get_feature_dimensions_demo(),
        'exam': 9,
        'other_disease': dataset.get_feature_dimensions_od(),
        'hamd17': 1,
        'scale': 17,
    }

    num_task = dataset.get_task_num()
    model = ClinicalModelDynamic(modal_dims, drug_vocab_size=100, hidden_dim=256, n_heads=2, n_layers=2, num_classes=4, num_tasks=num_task).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def explain_model_with_shap_kernelexplainer(model, data_loader, device, save_dir, on, mode):
    batch = next(iter(data_loader))
    batch_inputs, _ = batch
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

    def model_predict(input_array, mode=mode):
        input_array = input_array.reshape(-1, 10, 229)
        # Convert input_array back to dictionary format
        inputs = {}
        for modal, tensor in batch_inputs.items():
            if modal == 'gene':
                inputs[modal] = torch.tensor(input_array[:, 0, :], dtype=torch.float32).to(device)
            elif modal == 'age':
                inputs[modal] = torch.tensor(input_array[:, 1, 0], dtype=torch.float32).to(device)
            elif modal == 'bmi':
                inputs[modal] = torch.tensor(input_array[:, 2, 0], dtype=torch.float32).to(device)
            elif modal == 'demo':
                inputs[modal] = torch.tensor(input_array[:, 3, :28], dtype=torch.float32).to(device)
            elif modal == 'duration':
                inputs[modal] = torch.tensor(input_array[:, 4, 0], dtype=torch.float32).to(device)
            elif modal == 'exam':
                inputs[modal] = torch.tensor(input_array[:, 5, :9], dtype=torch.float32).to(device)
            elif modal == 'scale':
                inputs[modal] = torch.tensor(input_array[:, 6, :17], dtype=torch.float32).to(device)
            elif modal == 'other_disease':
                inputs[modal] = torch.tensor(input_array[:, 7, :18], dtype=torch.float32).to(device)
            elif modal == 'hamd17':
                inputs[modal] = torch.tensor(input_array[:, 8, 0], dtype=torch.float32).unsqueeze(1).unsqueeze(1).to(device)
            elif modal == 'medication_history':
                inputs[modal] = torch.tensor(input_array[:, 9, :40], dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            
        # Convert outputs to numpy array
        if isinstance(outputs, list):
            if mode == 'argmax':
                return np.array([output.cpu().numpy() for output in outputs]).argmax(axis=-1)[on]
            elif mode == '3':
                return np.array([output.cpu().numpy() for output in outputs])[:,:,3][on]
        else:
            return outputs.cpu().numpy()

    # Convert batch_inputs to a single numpy array for KernelExplainer
    def pad_tensor(tensor, target_shape):
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(1).unsqueeze(1)
        elif len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(1)
            
        pad_size = target_shape[-1] - tensor.shape[-1]
        return torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0)

    batch_inputs_padded = {k: pad_tensor(v, (v.shape[0], 229)) for k, v in batch_inputs.items()}
    input_array = np.concatenate([v.cpu().numpy() for v in batch_inputs_padded.values()], axis=1)
    input_array = input_array.reshape(input_array.shape[0], -1)
    explainer = shap.Explainer(model_predict, input_array)
    shap_values = explainer(input_array, max_evals=1000)
    
    shap.summary_plot(shap_values, input_array, max_display=20)
    plt.savefig(f'{save_dir}/shap_summary_plot_test_drug{on}_{mode}_test.png')
    joblib.dump(shap_values, f'{save_dir}/shap_values_drug{on}_{mode}_test.joblib')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shaply analysis')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--on', type=int, default=0, help='perform shap analysis on which dimension')
    parser.add_argument('--save_dir', type=str, default='./shap_results', help='Directory to save SHAP results')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, train_loader, test_dataset, test_loader = load_data(args.data_path, random_seed=248)
    model = load_model(model_path=args.model_path, dataset=test_dataset, device=device)

    explain_model_with_shap_kernelexplainer(model, test_loader, device, save_dir=args.save_dir, on=args.on, mode='argmax')
    plt.close()

