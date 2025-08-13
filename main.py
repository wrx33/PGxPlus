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
from datetime import datetime
import argparse
from data import *
from model import ClinicalModelDynamic
from loss import *
from shap_explain import *
from config import sweep_config
import os
import wandb
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def compute_multi_task_loss(outputs, labels, criterion):
    total_loss = 0
    for i in range(len(outputs)):
        total_loss += criterion(outputs[i], labels[:, i])
    return total_loss

def train_model(model, train_loader, criterion, optimizer, device, clip_value=1.0):
    model.train()
    total_loss = 0
    for batch_inputs, labels in train_loader:
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = compute_multi_task_loss(outputs, labels, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device, verbose=False):
    model.eval()
    correct_quar = 0
    correct_bina = 0
    total = 0
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
    if verbose:
        gt = labels[:, i].cpu().numpy().astype(int)
        print(f'Predition   : {predicted.cpu().numpy()} \nGround Truth: {gt}')
    return correct_quar / total, correct_bina / total

class EarlyStopping:
    def __init__(self, patience=10, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.epochs_without_improvement = 0
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        return self.epochs_without_improvement >= self.patience

def main():
    parser = argparse.ArgumentParser(description='Train Clinical Model')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--lr', type=float, default=0.00039107813754462944, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default='your_data_path', help='Dataset path')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save models')
    args = parser.parse_args()

    hidden_dim = args.hidden_dim
    n_heads = args.n_heads
    n_layers = args.n_layers
    num_classes = args.num_classes
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = f'./{args.save_dir}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(project="pgx_stage3_0122")

    data = torch.load(args.dataset)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = ClinicalDataset_simple(train_data)
    test_dataset = ClinicalDataset_simple(test_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    modal_dims = {
        'gene': train_dataset.get_feature_dimensions_gene(),
        'age': 1,
        'demo': train_dataset.get_feature_dimensions_demo(),
        'duration': 1,
        'exam': 7,
        'scale': 24,
        'description1': 1,
        'description2': 1,
        'description3': 1,
        'description4': 128,
        'medication_history': 13
    }

    num_task = train_dataset.get_task_num()

    model = ClinicalModelDynamic(modal_dims, drug_vocab_size=100, hidden_dim=args.hidden_dim, n_heads=args.n_heads, n_layers=args.n_layers, num_classes=num_classes, num_tasks=num_task).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    class_weights = torch.tensor([10.0, 10.0, 1.0, 1.0], dtype=torch.float32)
    criterion = BalancedLoss(class_weights, alpha=1.0, gamma=2.0, lambda_ce=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
    best_test_acc = 0
    wandb.watch(model, log="all")  
    early_stopping = EarlyStopping(patience=300, delta=0.001)

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        train_acc_quar, train_acc_bina = evaluate_model(model, train_loader, device)
        if epoch % 50 == 0:
            test_acc_quar, test_acc_bina = evaluate_model(model, test_loader, device, verbose=True)
        else:
            test_acc_quar, test_acc_bina = evaluate_model(model, test_loader, device)
        
        scheduler.step(train_loss)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc Quarter: {train_acc_quar:.4f}, \
              Train Acc Binary: {train_acc_bina:.4f}, Test Acc Quarter: {test_acc_quar:.4f}, Test Acc Binary: {test_acc_bina:.4f}')
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc_quar": train_acc_quar,
            "test_acc_quar": test_acc_quar,
            "train_acc_bina": train_acc_bina,
            "test_acc_bina": test_acc_bina,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        if epoch == 0 or test_acc_quar > best_test_acc:
            best_test_acc = test_acc_quar
            torch.save(model.state_dict(), f'./{save_dir}/best_model.pth')
            wandb.save(f'./{save_dir}/best_model.pth')

        if early_stopping(test_acc_quar):
            print(f'Early stopping at epoch {epoch}')
            break

if __name__ == '__main__':
    main()
    
        