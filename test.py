import torch
import torch.nn as nn
from data import *
from model import ClinicalModelDynamic
from loss import *
from shap_explain import *
from config import sweep_config
import json
import os
import argparse
import wandb
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_model(model_path, dataset, device):
    modal_dims = {
        'gene': dataset.get_feature_dimensions_gene(),  #229
        'age': 1,
        'bmi': 1,
        'demo': dataset.get_feature_dimensions_demo(),  # 28
        'duration': 1,
        'exam': 9,
        'scale': 17,
        'other_disease': dataset.get_feature_dimensions_od(),   # 18
        'hamd17': 1,
        'medication_history':  dataset.get_feature_dimensions_mh(), # 40
    }

    num_task = dataset.get_task_num()
    model = ClinicalModelDynamic(modal_dims, drug_vocab_size=100, hidden_dim=256, n_heads=2, n_layers=2, num_classes=4, num_tasks=num_task).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_data(data_path, random_seed):
    data = torch.load(data_path)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)
    train_dataset = ClinicalDataset_simple(train_data)
    test_dataset = ClinicalDataset_simple(test_data)
    train_loader = DataLoader(train_dataset, batch_size=2500, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2500, shuffle=False)
    return train_dataset, train_loader, test_dataset, test_loader

def evaluate_model(model, test_loader, device, verbose=False, save_tag=None):
    correct_quar = 0
    correct_bina = 0
    total = 0
    predictions = []
    groundtruths = []
    with torch.no_grad():
        for batch_inputs, labels in test_loader:
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            labels = labels.to(device)
            outputs = model(batch_inputs)
            for i in range(len(outputs)):
                outputs[i] = F.softmax(outputs[i], dim=1)
                outputs[i] = torch.argmax(outputs[i], dim=1)
                predicted = torch.round(outputs[i])
                total += labels.size(0)
                
                predictions.extend(predicted.cpu().numpy())
                groundtruths.extend(labels[:, i].cpu().numpy())
                correct_quar += (predicted == labels[:, i]).sum().item()

                predicted = predicted.clone()
                predicted[predicted == 1] = 0
                predicted[predicted == 2] = 1
                predicted[predicted == 3] = 1
                labels[:, i] = labels[:, i].clone()
                labels[:, i][labels[:, i] == 1] = 0
                labels[:, i][labels[:, i] == 2] = 1
                labels[:, i][labels[:, i] == 3] = 1
                
                correct_bina += (predicted == labels[:, i]).sum().item()

    print(f'Predition   : {predictions} \nGround Truth: {groundtruths}')    
    step = int(len(predictions)/len(outputs))
    num_lists = step 
    predictions_drugs = []
    groundtruths_drugs = []
    for i in range(num_lists):
        predictions_drugs.append(predictions[i::step])
        groundtruths_drugs.append(list(map(int, groundtruths[i::step])))

    output_data = {
        "predictions_drugs": [[int(x) for x in sublist] for sublist in predictions_drugs],
        "groundtruths_drugs": [[int(x) for x in sublist] for sublist in groundtruths_drugs]
    }

    output_file = f"/root/output_data_{save_tag}.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f'Four-category accuracy: {(correct_quar / total):.4f}, Binary accuracy: {(correct_bina / total):.4f}')
    return predictions, groundtruths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input and output paths.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for data splitting.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file.")
    parser.add_argument("--save_tag", type=str, default="test", help="Tag for the output file.")
    args = parser.parse_args()

    random_seed = args.random_seed
    model_path = args.model_path
    save_tag = args.save_tag
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, train_loader, test_dataset, test_loader = load_data(args.data_path, random_seed=random_seed)
    model = load_model(model_path=model_path, dataset=test_dataset, device=device)
    predictions, groundtruths = evaluate_model(model, test_loader, device, save_tag=save_tag)
    